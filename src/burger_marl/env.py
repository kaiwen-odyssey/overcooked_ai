"""Authoritative burger task dynamics for multi-agent training.

The standalone transition order is designed for the commercial gameplay loop:

1. validate the complete joint action;
2. resolve adjacent PICK_DROP and PROCESS actions by player index;
3. resolve all movement and collisions simultaneously;
4. advance autonomous environment timers exactly once.

The renderer is not part of this module. Rewards are derived exclusively from
events emitted by a validated state transition, so a policy cannot claim an
event or modify a displayed score.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from burger_marl.actions import Action, Direction

Point = Tuple[int, int]
Item = str

COUNTER = "X"
FLOOR = " "
BUN_DISPENSER = "B"
LETTUCE_DISPENSER = "L"
BEEF_DISPENSER = "M"
GRILL = "G"
PLATE_RACK = "D"
SINK = "W"
SERVE = "S"
RETURN = "R"
EXTINGUISHER = "E"
TRASH = "T"

TERRAIN_TYPES = {
    COUNTER,
    FLOOR,
    BUN_DISPENSER,
    LETTUCE_DISPENSER,
    BEEF_DISPENSER,
    GRILL,
    PLATE_RACK,
    SINK,
    SERVE,
    RETURN,
    EXTINGUISHER,
    TRASH,
}
VALID_ITEMS = {
    "bun",
    "lettuce",
    "raw_beef",
    "cooked_beef",
    "clean_plate",
    "plate_bun",
    "plate_lettuce",
    "plate_cooked_beef",
    "plate_bun_lettuce",
    "plate_bun_cooked_beef",
    "plate_lettuce_cooked_beef",
    "dirty_plate",
    "plated_burger",
    "extinguisher",
}
PLATE_CONTENTS = {
    "clean_plate": frozenset(),
    "plate_bun": frozenset({"bun"}),
    "plate_lettuce": frozenset({"lettuce"}),
    "plate_cooked_beef": frozenset({"cooked_beef"}),
    "plate_bun_lettuce": frozenset({"bun", "lettuce"}),
    "plate_bun_cooked_beef": frozenset({"bun", "cooked_beef"}),
    "plate_lettuce_cooked_beef": frozenset({"lettuce", "cooked_beef"}),
    "plated_burger": frozenset({"bun", "lettuce", "cooked_beef"}),
}
CONTENTS_TO_PLATE = {
    contents: item for item, contents in PLATE_CONTENTS.items()
}
PLATE_ITEMS = set(PLATE_CONTENTS) | {"dirty_plate"}
# Keep every non-terminal potential non-negative. This preserves the
# telescoping shaping contract while ensuring a finite-rollout state cycle
# cannot profit merely by lingering in a negative-potential recovery state.
POTENTIAL_FLOOR = 18.0


@dataclass(frozen=True)
class BurgerLayout:
    """Static terrain and player spawn contract."""

    rows: Tuple[str, ...]
    player_starts: Tuple[Point, ...]
    name: str = "burger_layout"

    def __post_init__(self) -> None:
        if not self.rows or not self.rows[0]:
            raise ValueError("Layout must contain at least one cell")
        width = len(self.rows[0])
        if any(len(row) != width for row in self.rows):
            raise ValueError("Layout rows must have equal width")
        invalid = {
            cell for row in self.rows for cell in row if cell not in TERRAIN_TYPES
        }
        if invalid:
            raise ValueError("Unknown terrain symbols: {}".format(sorted(invalid)))
        if not 1 <= len(self.player_starts) <= 4:
            raise ValueError("Burger task supports one to four players")
        if len(set(self.player_starts)) != len(self.player_starts):
            raise ValueError("Player starts must be unique")
        if any(self.terrain_at(pos) != FLOOR for pos in self.player_starts):
            raise ValueError("Every player must start on walkable floor")
        required = {
            BUN_DISPENSER,
            LETTUCE_DISPENSER,
            BEEF_DISPENSER,
            GRILL,
            PLATE_RACK,
            SINK,
            SERVE,
            RETURN,
            EXTINGUISHER,
            TRASH,
        }
        present = set("".join(self.rows))
        missing = required - present
        if missing:
            raise ValueError(
                "Burger layout missing required stations: {}".format(sorted(missing))
            )

    @cached_property
    def width(self) -> int:
        return len(self.rows[0])

    @cached_property
    def height(self) -> int:
        return len(self.rows)

    def terrain_at(self, point: Point) -> Optional[str]:
        x, y = point
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return None
        return self.rows[y][x]

    @cached_property
    def valid_player_positions(self) -> frozenset[Point]:
        return frozenset(
            (x, y)
            for y, row in enumerate(self.rows)
            for x, terrain in enumerate(row)
            if terrain == FLOOR
        )


CRAMPED_GALLEY = BurgerLayout(
    name="cramped_galley",
    rows=(
        "BLXMXGET",
        "        ",
        "        ",
        "XX XXX X",
        "   XX   ",
        "        ",
        "        ",
        "WDXXXXRS",
    ),
    player_starts=((0, 1), (3, 1), (1, 6), (7, 6)),
)

CURRICULUM_START_STAGES = (
    "standard",
    "serve_ready",
    "grill_ready",
    "cook_ready",
    "plate_ready",
    "plate_pick_ready",
    "rack_ready",
    "raw_ready",
    "grill_approach_ready",
    "beef_carry_ready",
    "beef_pick_ready",
    "wash_ready",
    "wash_required_ready",
    "sink_drop_ready",
    "dirty_plate_carry_ready",
    "dirty_return_ready",
    "second_dish_ready",
    "plate_exhausted_ready",
    "fire_recovery_ready",
    "fire_plate_drop_ready",
    "fire_plate_parked_ready",
    "fire_extinguisher_pick_ready",
    "fire_extinguisher_carry_ready",
    "fire_suppress_ready",
)


@dataclass
class BurgerPlayerState:
    position: Point
    orientation: Point
    held_object: Optional[Item] = None


@dataclass
class GrillState:
    food: Optional[str] = None
    cook_ticks: int = 0
    ready_ticks: int = 0


@dataclass
class SinkState:
    has_dirty_plate: bool = False
    wash_progress: int = 0


@dataclass
class BurgerState:
    players: List[BurgerPlayerState]
    counter_objects: Dict[Point, Item] = field(default_factory=dict)
    timestep: int = 0
    clean_plates: int = 4
    dirty_plates_at_return: int = 0
    pending_plate_returns: List[int] = field(default_factory=list)
    total_plates: int = 4
    grill: GrillState = field(default_factory=GrillState)
    sink: SinkState = field(default_factory=SinkState)
    delivered_orders: int = 0
    extinguisher_available: bool = True


@dataclass(frozen=True)
class BurgerRewardConfig:
    correct_delivery: float = 20.0
    # Curriculum milestone events remain observable in the reward breakdown,
    # but the final task objective gives them no standalone positive reward.
    # Learning guidance comes from policy-invariant potential shaping instead.
    raw_beef_placed: float = 0.0
    dirty_plate_pickup: float = 0.0
    wash_started: float = 0.0
    wash_progress: float = 0.0
    plate_washed: float = 0.0
    fire_extinguished: float = 0.0
    fire_started: float = -5.0
    # Charge for every transition that ends with the grill still burning.
    # Extinguishing immediately therefore pays no recurring cost, while each
    # delayed control step is strictly worse. Unlike an extinguish bonus this
    # signal cannot be farmed by repeatedly creating and clearing fires.
    fire_active: float = -0.25
    # Safety cost only: while the grill is burning, touching ingredient
    # dispensers delays the extinguisher response and can strand unusable raw
    # food on a counter. The official gameplay action remains legal.
    fire_food_handling: float = -2.0
    # Parking a dirty plate can be legal (for example to answer a fire), but
    # repeated counter drop/pickup cycles waste throughput. A small handling
    # cost breaks that loop without forbidding the gameplay action.
    dirty_plate_counter_handling: float = -0.25
    collision: float = -0.05
    # A constant step cost cannot change a fixed-horizon throughput objective.
    # Keep it zero so the unshaped return is dominated by real deliveries.
    time_step: float = 0.0
    potential_scale: float = 1.0
    navigation_potential_scale: float = 1.0
    # Match the long-horizon PPO default so one-cell navigation and one-tick
    # work progress remain positive after bounded potential shaping.
    gamma: float = 0.999


@dataclass(frozen=True)
class BurgerConfig:
    # At 0.42 seconds per control step: 10.08 seconds to cook, followed by
    # a 6.72-second pickup window before the grill catches fire.
    cook_steps: int = 24
    burn_steps: int = 16
    wash_steps: int = 10
    # A served plate is out of the kitchen for 5.04 seconds before it re-enters
    # at the dedicated dish-return hatch.
    plate_return_steps: int = 12
    # 180 seconds at the renderer/trainer control period of 0.42 seconds.
    horizon: int = 429
    total_plates: int = 4
    # False permits following a teammate into a cell they vacate on the same
    # joint step. The stricter occupied-at-step-start variant is an ablation.
    strict_agent_obstacles: bool = False
    reward: BurgerRewardConfig = field(default_factory=BurgerRewardConfig)

    def __post_init__(self) -> None:
        positive_fields = {
            "cook_steps": self.cook_steps,
            "burn_steps": self.burn_steps,
            "wash_steps": self.wash_steps,
            "plate_return_steps": self.plate_return_steps,
            "horizon": self.horizon,
            "total_plates": self.total_plates,
        }
        invalid = [name for name, value in positive_fields.items() if value <= 0]
        if invalid:
            raise ValueError(
                "Configuration values must be positive: {}".format(invalid)
            )


@dataclass(frozen=True)
class BurgerTransition:
    state: BurgerState
    reward: float
    done: bool
    info: Mapping[str, object]


class BurgerGridworld:
    """Pure, deterministic burger dynamics with invariant checks."""

    def __init__(
        self,
        layout: BurgerLayout = CRAMPED_GALLEY,
        config: BurgerConfig = BurgerConfig(),
    ) -> None:
        self.layout = layout
        self.config = config
        self._station_distances = {
            terrain: self._distance_field_for_station(terrain)
            for terrain in (
                BUN_DISPENSER,
                LETTUCE_DISPENSER,
                BEEF_DISPENSER,
                GRILL,
                PLATE_RACK,
                SINK,
                SERVE,
                RETURN,
                EXTINGUISHER,
            )
        }
        self._counter_distances = {
            (x, y): self._distance_field_for_position((x, y))
            for y, row in enumerate(self.layout.rows)
            for x, terrain in enumerate(row)
            if terrain == COUNTER
        }

    def _distance_field_for_station(
        self, terrain: str
    ) -> Dict[Point, int]:
        stations = [
            (x, y)
            for y, row in enumerate(self.layout.rows)
            for x, cell in enumerate(row)
            if cell == terrain
        ]
        if len(stations) != 1:
            return {}
        return self._distance_field_for_position(stations[0])

    def _distance_field_for_position(
        self, station: Point
    ) -> Dict[Point, int]:
        goals = [
            Action.move_in_direction(station, direction)
            for direction in Direction.ALL_DIRECTIONS
            if self.layout.terrain_at(
                Action.move_in_direction(station, direction)
            )
            == FLOOR
        ]
        distances = {goal: 0 for goal in goals}
        queue = deque(goals)
        while queue:
            position = queue.popleft()
            for direction in Direction.ALL_DIRECTIONS:
                neighbor = Action.move_in_direction(position, direction)
                if (
                    neighbor not in distances
                    and self.layout.terrain_at(neighbor) == FLOOR
                ):
                    distances[neighbor] = distances[position] + 1
                    queue.append(neighbor)
        return distances

    def get_standard_start_state(self, num_players: int = 4) -> BurgerState:
        if not 1 <= num_players <= len(self.layout.player_starts):
            raise ValueError("num_players exceeds layout capacity")
        players = [
            BurgerPlayerState(position=pos, orientation=Direction.NORTH)
            for pos in self.layout.player_starts[:num_players]
        ]
        state = BurgerState(
            players=players,
            clean_plates=self.config.total_plates,
            total_plates=self.config.total_plates,
        )
        self.validate_state(state)
        return state

    def get_start_state(
        self,
        num_players: int = 4,
        stage: str = "standard",
    ) -> BurgerState:
        """Return a legal reset state for direct-PPO reverse curriculum."""

        if stage not in CURRICULUM_START_STAGES:
            raise ValueError("Unknown curriculum start stage: {!r}".format(stage))
        state = self.get_standard_start_state(num_players)
        if stage == "standard":
            return state
        if num_players != 1:
            raise ValueError(
                "Curriculum start states support one active agent only"
            )

        def station_position(terrain: str) -> Point:
            matches = [
                (x, y)
                for y, row in enumerate(self.layout.rows)
                for x, cell in enumerate(row)
                if cell == terrain
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Curriculum requires one {!r} station".format(terrain)
                )
            return matches[0]

        def adjacent_floor(terrain: str) -> Point:
            target = station_position(terrain)
            for direction in Direction.ALL_DIRECTIONS:
                candidate = Action.move_in_direction(target, direction)
                if self.layout.terrain_at(candidate) == FLOOR:
                    return candidate
            raise ValueError(
                "Curriculum station {!r} has no adjacent floor".format(terrain)
            )

        def counter_with_adjacent_floor(
            nearest_to: Optional[str] = None,
        ) -> Tuple[Point, Point]:
            candidates: List[Tuple[Point, Point]] = []
            for y, row in enumerate(self.layout.rows):
                for x, terrain in enumerate(row):
                    if terrain != COUNTER:
                        continue
                    counter = (x, y)
                    for direction in Direction.ALL_DIRECTIONS:
                        candidate = Action.move_in_direction(
                            counter, direction
                        )
                        if self.layout.terrain_at(candidate) == FLOOR:
                            candidates.append((counter, candidate))
                            break
            if candidates:
                if nearest_to is None:
                    return candidates[0]
                target = station_position(nearest_to)
                return min(
                    candidates,
                    key=lambda item: (
                        abs(item[0][0] - target[0])
                        + abs(item[0][1] - target[1]),
                        item,
                    ),
                )
            raise ValueError(
                "Curriculum requires a counter beside walkable floor"
            )

        player = state.players[0]
        if stage == "serve_ready":
            player.position = adjacent_floor(SERVE)
            player.held_object = "plated_burger"
            state.clean_plates -= 1
        elif stage in {"grill_ready", "cook_ready"}:
            player.position = adjacent_floor(GRILL)
            player.held_object = "plate_bun_lettuce"
            state.clean_plates -= 1
            state.grill = GrillState(
                food=(
                    "cooked_beef"
                    if stage == "grill_ready"
                    else "raw_beef"
                ),
                cook_ticks=(
                    self.config.cook_steps
                    if stage == "grill_ready"
                    else 0
                ),
                ready_ticks=0,
            )
        elif stage == "plate_ready":
            player.position = adjacent_floor(PLATE_RACK)
            player.held_object = "clean_plate"
            state.clean_plates -= 1
            state.grill = GrillState(
                food="raw_beef",
                cook_ticks=0,
                ready_ticks=0,
            )
        elif stage == "plate_pick_ready":
            player.position = adjacent_floor(PLATE_RACK)
            state.grill = GrillState(
                food="raw_beef",
                cook_ticks=0,
                ready_ticks=0,
            )
        elif stage == "rack_ready":
            player.position = adjacent_floor(GRILL)
            state.grill = GrillState(
                food="raw_beef",
                cook_ticks=0,
                ready_ticks=0,
            )
        elif stage == "raw_ready":
            player.position = adjacent_floor(GRILL)
            player.held_object = "raw_beef"
        elif stage == "grill_approach_ready":
            candidates = [
                position
                for position, distance in self._station_distances[
                    GRILL
                ].items()
                if distance == 1
            ]
            if not candidates:
                raise ValueError(
                    "Curriculum grill has no one-step approach cell"
                )
            player.position = min(
                candidates,
                key=lambda position: self._station_distances[
                    BEEF_DISPENSER
                ].get(position, self.layout.width * self.layout.height),
            )
            player.held_object = "raw_beef"
        elif stage == "beef_carry_ready":
            player.position = adjacent_floor(BEEF_DISPENSER)
            player.held_object = "raw_beef"
        elif stage == "beef_pick_ready":
            player.position = adjacent_floor(BEEF_DISPENSER)
        elif stage == "wash_ready":
            player.position = adjacent_floor(SINK)
            state.clean_plates -= 1
            state.sink = SinkState(has_dirty_plate=True, wash_progress=0)
        elif stage == "wash_required_ready":
            player.position = adjacent_floor(SINK)
            state.clean_plates = 0
            state.dirty_plates_at_return = state.total_plates - 1
            state.sink = SinkState(has_dirty_plate=True, wash_progress=0)
            state.delivered_orders = state.total_plates
            state.grill = GrillState(
                food="raw_beef",
                cook_ticks=0,
                ready_ticks=0,
            )
        elif stage in {"sink_drop_ready", "dirty_plate_carry_ready"}:
            player.position = adjacent_floor(
                SINK
                if stage == "sink_drop_ready"
                else RETURN
            )
            player.held_object = "dirty_plate"
            state.clean_plates = 0
            state.dirty_plates_at_return = state.total_plates - 1
            state.delivered_orders = state.total_plates
            state.grill = GrillState(
                food="raw_beef",
                cook_ticks=0,
                ready_ticks=0,
            )
        elif stage == "dirty_return_ready":
            player.position = adjacent_floor(RETURN)
            state.clean_plates -= 1
            state.dirty_plates_at_return = 1
        elif stage == "second_dish_ready":
            player.position = adjacent_floor(RETURN)
            state.clean_plates -= 1
            state.dirty_plates_at_return = 1
            state.delivered_orders = 1
            state.grill = GrillState(
                food="raw_beef",
                cook_ticks=0,
                ready_ticks=0,
            )
        elif stage == "plate_exhausted_ready":
            player.position = adjacent_floor(RETURN)
            state.clean_plates = 0
            state.dirty_plates_at_return = state.total_plates
            state.delivered_orders = state.total_plates
            state.grill = GrillState(
                food="raw_beef",
                cook_ticks=0,
                ready_ticks=0,
            )
        elif stage == "fire_recovery_ready":
            player.position = adjacent_floor(EXTINGUISHER)
            player.held_object = "clean_plate"
            state.clean_plates -= 1
            state.grill = GrillState(food="burnt_beef")
        elif stage in {
            "fire_plate_drop_ready",
            "fire_plate_parked_ready",
            "fire_extinguisher_pick_ready",
            "fire_extinguisher_carry_ready",
            "fire_suppress_ready",
        }:
            counter, beside_counter = counter_with_adjacent_floor(
                GRILL if stage == "fire_plate_parked_ready" else None
            )
            state.clean_plates -= 1
            state.grill = GrillState(food="burnt_beef")
            if stage == "fire_plate_drop_ready":
                player.position = beside_counter
                player.held_object = "clean_plate"
            elif stage == "fire_plate_parked_ready":
                player.position = beside_counter
                state.counter_objects[counter] = "plate_bun_lettuce"
            else:
                state.counter_objects[counter] = "clean_plate"
                if stage == "fire_extinguisher_pick_ready":
                    player.position = adjacent_floor(EXTINGUISHER)
                elif stage == "fire_extinguisher_carry_ready":
                    player.position = adjacent_floor(EXTINGUISHER)
                    player.held_object = "extinguisher"
                    state.extinguisher_available = False
                else:
                    player.position = adjacent_floor(GRILL)
                    player.held_object = "extinguisher"
                    state.extinguisher_available = False
        self.validate_state(state)
        return state

    def get_state_transition(
        self, state: BurgerState, joint_action: Sequence[object]
    ) -> BurgerTransition:
        self.validate_state(state)
        if state.timestep >= self.config.horizon:
            raise ValueError("Cannot step a terminal burger state")
        self._validate_joint_action(state, joint_action)

        old_phi = self.potential(state)
        new_state = copy.deepcopy(state)
        events: List[Dict[str, object]] = []

        self._resolve_interacts(new_state, joint_action, events)
        if state.grill.food == "burnt_beef":
            unrelated_food_events = {
                "dispenser_pickup",
                "ingredient_added_from_dispenser",
            }
            events.extend(
                {
                    "type": "fire_food_handling",
                    "agent": event.get("agent"),
                    "source_event": event["type"],
                }
                for event in tuple(events)
                if event["type"] in unrelated_food_events
            )
        events.extend(
            {
                "type": "dirty_plate_counter_handling",
                "agent": event.get("agent"),
                "source_event": event["type"],
            }
            for event in tuple(events)
            if event["type"] in {"counter_drop", "counter_pickup"}
            and event.get("item") == "dirty_plate"
        )
        self._resolve_movement(new_state, joint_action, events)
        self._step_environment_effects(new_state, events)
        if new_state.grill.food == "burnt_beef":
            events.append({"type": "fire_active"})
        self.validate_state(new_state)

        done = new_state.timestep >= self.config.horizon
        new_phi = 0.0 if done else self.potential(new_state)
        breakdown = self._reward_breakdown(events, old_phi, new_phi)
        reward = sum(breakdown.values())
        info: Mapping[str, object] = {
            "events": tuple(copy.deepcopy(events)),
            "reward_breakdown": breakdown,
            "sparse_reward": breakdown["correct_delivery"],
            "shaped_reward": breakdown["potential"],
            "state_digest": self.state_digest(new_state),
        }
        return BurgerTransition(new_state, reward, done, info)

    def _validate_joint_action(
        self, state: BurgerState, joint_action: Sequence[object]
    ) -> None:
        if len(joint_action) != len(state.players):
            raise ValueError(
                "Expected {} actions, got {}".format(
                    len(state.players), len(joint_action)
                )
            )
        for action in joint_action:
            if action not in Action.ALL_ACTIONS:
                raise ValueError("Illegal burger action: {!r}".format(action))

    def _resolve_interacts(
        self,
        state: BurgerState,
        joint_action: Sequence[object],
        events: List[Dict[str, object]],
    ) -> None:
        # Context actions use stable player-index order. Every later action sees
        # mutations made by earlier players in the same joint step.
        for player_idx, (player, action) in enumerate(
            zip(state.players, joint_action)
        ):
            if action not in {Action.PICK_DROP, Action.PROCESS}:
                continue
            target = self._context_action_target(
                state, player_idx, action
            )
            if target is None:
                continue
            terrain = self.layout.terrain_at(target)

            if action == Action.PROCESS:
                if terrain == SINK:
                    self._process_sink(state, player_idx, events)
                elif terrain == GRILL:
                    self._process_grill(state, player_idx, events)
                continue

            if terrain == COUNTER:
                self._interact_counter(state, player_idx, target, events)
            elif terrain == BUN_DISPENSER:
                self._take_from_dispenser(
                    player, player_idx, "bun", BUN_DISPENSER, events
                )
            elif terrain == LETTUCE_DISPENSER:
                self._take_from_dispenser(
                    player, player_idx, "lettuce", LETTUCE_DISPENSER, events
                )
            elif terrain == BEEF_DISPENSER:
                self._take_from_dispenser(
                    player, player_idx, "raw_beef", BEEF_DISPENSER, events
                )
            elif terrain == EXTINGUISHER:
                self._interact_extinguisher_station(
                    state, player_idx, events
                )
            elif terrain == PLATE_RACK:
                self._interact_plate_rack(state, player_idx, events)
            elif terrain == GRILL:
                self._interact_grill(state, player_idx, events)
            elif terrain == SERVE:
                self._interact_serve(state, player_idx, events)
            elif terrain == RETURN:
                self._interact_return(state, player_idx, events)
            elif terrain == SINK:
                self._pick_drop_sink(state, player_idx, events)
            elif terrain == TRASH:
                self._interact_trash(state, player_idx, events)

    def _interact_counter(
        self,
        state: BurgerState,
        player_idx: int,
        target: Point,
        events: List[Dict[str, object]],
    ) -> None:
        """Atomically transform, pick up, or fill one single-slot counter."""

        player = state.players[player_idx]
        counter_item = state.counter_objects.get(target)
        combined = self._combine_plate_and_ingredient(
            player.held_object, counter_item
        )
        if combined is not None:
            player.held_object = combined
            state.counter_objects.pop(target)
            events.append(
                {
                    "type": "ingredient_added_to_held_plate",
                    "agent": player_idx,
                    "item": counter_item,
                    "position": target,
                }
            )
            return

        combined = self._combine_plate_and_ingredient(
            counter_item, player.held_object
        )
        if combined is not None:
            state.counter_objects[target] = combined
            events.append(
                {
                    "type": "ingredient_added_to_counter_plate",
                    "agent": player_idx,
                    "item": player.held_object,
                    "position": target,
                }
            )
            player.held_object = None
            return

        if player.held_object is not None and target not in state.counter_objects:
            state.counter_objects[target] = player.held_object
            events.append(
                {
                    "type": "counter_drop",
                    "agent": player_idx,
                    "item": player.held_object,
                    "position": target,
                }
            )
            player.held_object = None
        elif (
            player.held_object is None
            and target in state.counter_objects
            and counter_item != "cooked_beef"
        ):
            player.held_object = state.counter_objects.pop(target)
            events.append(
                {
                    "type": "counter_pickup",
                    "agent": player_idx,
                    "item": player.held_object,
                    "position": target,
                }
            )

    def _take_from_dispenser(
        self,
        player: BurgerPlayerState,
        player_idx: int,
        item: Item,
        station: str,
        events: List[Dict[str, object]],
    ) -> None:
        combined = self._combine_plate_and_ingredient(
            player.held_object, item
        )
        if combined is not None:
            player.held_object = combined
            events.append(
                {
                    "type": "ingredient_added_from_dispenser",
                    "agent": player_idx,
                    "item": item,
                    "station": station,
                }
            )
            return
        if player.held_object is not None:
            return
        player.held_object = item
        events.append(
            {
                "type": "dispenser_pickup",
                "agent": player_idx,
                "item": item,
                "station": station,
            }
        )

    def _interact_plate_rack(
        self,
        state: BurgerState,
        player_idx: int,
        events: List[Dict[str, object]],
    ) -> None:
        player = state.players[player_idx]
        if player.held_object is None and state.clean_plates > 0:
            state.clean_plates -= 1
            player.held_object = "clean_plate"
            events.append({"type": "clean_plate_pickup", "agent": player_idx})
        elif player.held_object == "clean_plate":
            player.held_object = None
            state.clean_plates += 1
            events.append({"type": "clean_plate_return", "agent": player_idx})

    def _interact_extinguisher_station(
        self,
        state: BurgerState,
        player_idx: int,
        events: List[Dict[str, object]],
    ) -> None:
        player = state.players[player_idx]
        if player.held_object is None and state.extinguisher_available:
            player.held_object = "extinguisher"
            state.extinguisher_available = False
            events.append(
                {"type": "extinguisher_pickup", "agent": player_idx}
            )
        elif (
            player.held_object == "extinguisher"
            and not state.extinguisher_available
        ):
            player.held_object = None
            state.extinguisher_available = True
            events.append(
                {"type": "extinguisher_return", "agent": player_idx}
            )

    def context_action_available(
        self,
        state: BurgerState,
        player_idx: int,
        action: str,
    ) -> bool:
        """Whether an adjacent context action can produce an agent event.

        This predicate is the authoritative action-mask source. It only uses
        state that is represented in the actor's local observation and
        never exposes reward, distant inventory, or teammate intent.
        """

        if action not in {Action.PICK_DROP, Action.PROCESS}:
            raise ValueError("Expected PICK_DROP or PROCESS")
        return (
            self._context_action_target(state, player_idx, action)
            is not None
        )

    def _context_action_target(
        self,
        state: BurgerState,
        player_idx: int,
        action: str,
    ) -> Optional[Point]:
        """Select one legal cardinal neighbor in stable N/S/E/W order."""

        if action not in {Action.PICK_DROP, Action.PROCESS}:
            raise ValueError("Expected PICK_DROP or PROCESS")
        player = state.players[player_idx]
        for direction in Direction.ALL_DIRECTIONS:
            target = Action.move_in_direction(
                player.position, direction
            )
            if self._context_action_available_at(
                state, player_idx, action, target
            ):
                return target
        return None

    def _context_action_available_at(
        self,
        state: BurgerState,
        player_idx: int,
        action: str,
        target: Point,
    ) -> bool:
        """Validate one candidate target without reading orientation."""

        player = state.players[player_idx]
        terrain = self.layout.terrain_at(target)

        if action == Action.PROCESS:
            if terrain == SINK:
                return (
                    state.sink.has_dirty_plate
                    and player.held_object is None
                )
            if terrain == GRILL:
                return (
                    player.held_object == "extinguisher"
                    and state.grill.food == "burnt_beef"
                )
            return False

        held = player.held_object
        if terrain == COUNTER:
            counter_item = state.counter_objects.get(target)
            return (
                self._combine_plate_and_ingredient(held, counter_item)
                is not None
                or self._combine_plate_and_ingredient(counter_item, held)
                is not None
                or (held is not None and counter_item is None)
                or (
                    held is None
                    and counter_item is not None
                    and counter_item != "cooked_beef"
                )
            )
        if terrain in {
            BUN_DISPENSER,
            LETTUCE_DISPENSER,
            BEEF_DISPENSER,
        }:
            ingredient = {
                BUN_DISPENSER: "bun",
                LETTUCE_DISPENSER: "lettuce",
                BEEF_DISPENSER: "raw_beef",
            }[terrain]
            return (
                held is None
                or self._combine_plate_and_ingredient(
                    held, ingredient
                )
                is not None
            )
        if terrain == EXTINGUISHER:
            return (
                held is None and state.extinguisher_available
            ) or (
                held == "extinguisher"
                and not state.extinguisher_available
            )
        if terrain == PLATE_RACK:
            return (
                (held is None and state.clean_plates > 0)
                or held == "clean_plate"
            )
        if terrain == GRILL:
            return (
                (held == "raw_beef" and state.grill.food is None)
                or (
                    state.grill.food == "cooked_beef"
                    and self._combine_plate_and_ingredient(
                        held, "cooked_beef"
                    )
                    is not None
                )
            )
        if terrain == SERVE:
            return held == "plated_burger"
        if terrain == RETURN:
            return held is None and state.dirty_plates_at_return > 0
        if terrain == SINK:
            return (
                held == "dirty_plate"
                and not state.sink.has_dirty_plate
            )
        if terrain == TRASH:
            return (
                held in PLATE_CONTENTS
                and held != "clean_plate"
            ) or held in {"bun", "lettuce", "raw_beef", "cooked_beef"}
        return False

    def _interact_grill(
        self,
        state: BurgerState,
        player_idx: int,
        events: List[Dict[str, object]],
    ) -> None:
        player = state.players[player_idx]
        grill = state.grill
        if player.held_object == "raw_beef" and grill.food is None:
            player.held_object = None
            grill.food = "raw_beef"
            grill.cook_ticks = 0
            grill.ready_ticks = 0
            events.append({"type": "raw_beef_placed", "agent": player_idx})
        elif grill.food == "cooked_beef":
            combined = self._combine_plate_and_ingredient(
                player.held_object, "cooked_beef"
            )
            if combined is not None:
                player.held_object = combined
                grill.food = None
                grill.cook_ticks = 0
                grill.ready_ticks = 0
                events.append(
                    {
                        "type": "cooked_beef_added_to_held_plate",
                        "agent": player_idx,
                    }
                )

    @staticmethod
    def _process_grill(
        state: BurgerState,
        player_idx: int,
        events: List[Dict[str, object]],
    ) -> None:
        player = state.players[player_idx]
        grill = state.grill
        if player.held_object == "extinguisher" and grill.food == "burnt_beef":
            grill.food = None
            grill.cook_ticks = 0
            grill.ready_ticks = 0
            events.append({"type": "fire_extinguished", "agent": player_idx})

    @staticmethod
    def _interact_trash(
        state: BurgerState,
        player_idx: int,
        events: List[Dict[str, object]],
    ) -> None:
        """Discard food while preserving every physical plate.

        A plate containing food is emptied and stays in the player's hand as
        a clean plate. Dirty plates must still be washed, and the extinguisher
        must be returned to its dispenser; neither can be deleted here.
        """

        player = state.players[player_idx]
        held = player.held_object
        if held in PLATE_CONTENTS and held != "clean_plate":
            player.held_object = "clean_plate"
            events.append(
                {
                    "type": "plate_contents_discarded",
                    "agent": player_idx,
                    "item": held,
                }
            )
        elif held in {"bun", "lettuce", "raw_beef", "cooked_beef"}:
            player.held_object = None
            events.append(
                {
                    "type": "food_discarded",
                    "agent": player_idx,
                    "item": held,
                }
            )

    def _interact_serve(
        self,
        state: BurgerState,
        player_idx: int,
        events: List[Dict[str, object]],
    ) -> None:
        player = state.players[player_idx]
        if player.held_object == "plated_burger":
            player.held_object = None
            state.delivered_orders += 1
            state.pending_plate_returns.append(
                state.timestep + self.config.plate_return_steps
            )
            events.append({"type": "correct_delivery", "agent": player_idx})

    @staticmethod
    def _interact_return(
        state: BurgerState,
        player_idx: int,
        events: List[Dict[str, object]],
    ) -> None:
        player = state.players[player_idx]
        if player.held_object is None and state.dirty_plates_at_return > 0:
            state.dirty_plates_at_return -= 1
            player.held_object = "dirty_plate"
            events.append({"type": "dirty_plate_pickup", "agent": player_idx})

    def _pick_drop_sink(
        self,
        state: BurgerState,
        player_idx: int,
        events: List[Dict[str, object]],
    ) -> None:
        player = state.players[player_idx]
        sink = state.sink
        if player.held_object == "dirty_plate" and not sink.has_dirty_plate:
            player.held_object = None
            sink.has_dirty_plate = True
            sink.wash_progress = 0
            events.append({"type": "wash_started", "agent": player_idx})

    def _process_sink(
        self,
        state: BurgerState,
        player_idx: int,
        events: List[Dict[str, object]],
    ) -> None:
        sink = state.sink
        player = state.players[player_idx]
        if not sink.has_dirty_plate or player.held_object is not None:
            return
        if any(event["type"] == "wash_progress" for event in events):
            return

        # One joint step can add at most one wash tick, regardless of the
        # number of agents simultaneously interacting with the sink.
        sink.wash_progress += 1
        events.append(
            {
                "type": "wash_progress",
                "agent": player_idx,
                "progress": sink.wash_progress,
            }
        )
        if sink.wash_progress == self.config.wash_steps:
            sink.has_dirty_plate = False
            sink.wash_progress = 0
            player.held_object = "clean_plate"
            events.append({"type": "plate_washed", "agent": player_idx})

    def _resolve_movement(
        self,
        state: BurgerState,
        joint_action: Sequence[object],
        events: List[Dict[str, object]],
    ) -> None:
        old_positions = tuple(player.position for player in state.players)
        new_positions: List[Point] = []
        new_orientations: List[Point] = []

        for player, action in zip(state.players, joint_action):
            if action not in Action.MOTION_ACTIONS:
                new_positions.append(player.position)
                new_orientations.append(player.orientation)
                continue
            target = Action.move_in_direction(player.position, action)
            orientation = player.orientation if action == Action.STAY else action
            new_orientations.append(orientation)
            new_positions.append(
                target
                if target in self.layout.valid_player_positions
                else player.position
            )

        resolved_positions, collided_agents = self._resolve_agent_collisions(
            old_positions, tuple(new_positions)
        )
        if collided_agents:
            events.append(
                {
                    "type": "collision",
                    "agents": tuple(sorted(collided_agents)),
                }
            )

        for player, position, orientation in zip(
            state.players, resolved_positions, new_orientations
        ):
            player.position = position
            player.orientation = orientation

    def _resolve_agent_collisions(
        self,
        old_positions: Tuple[Point, ...],
        proposed_positions: Tuple[Point, ...],
    ) -> Tuple[Tuple[Point, ...], frozenset[int]]:
        """Cancel only agents involved in a lockstep position conflict.

        Resolution is iterated because canceling one move can turn that
        agent's old square into a new obstacle for a following agent. Unrelated
        agents continue moving in the same joint step.
        """

        resolved = list(proposed_positions)
        collided: set[int] = set()
        while True:
            newly_collided: set[int] = set()

            if self.config.strict_agent_obstacles:
                old_occupants = {
                    position: index
                    for index, position in enumerate(old_positions)
                }
                for index, (old, target) in enumerate(
                    zip(old_positions, resolved)
                ):
                    occupant = old_occupants.get(target)
                    if (
                        target != old
                        and occupant is not None
                        and occupant != index
                    ):
                        newly_collided.add(index)

            for left in range(len(old_positions)):
                for right in range(left + 1, len(old_positions)):
                    if (
                        resolved[left] != old_positions[left]
                        and resolved[right] != old_positions[right]
                        and resolved[left] == old_positions[right]
                        and resolved[right] == old_positions[left]
                    ):
                        newly_collided.update((left, right))

            by_endpoint: Dict[Point, List[int]] = {}
            for index, endpoint in enumerate(resolved):
                by_endpoint.setdefault(endpoint, []).append(index)
            for indices in by_endpoint.values():
                if len(indices) > 1:
                    newly_collided.update(
                        index
                        for index in indices
                        if resolved[index] != old_positions[index]
                    )

            newly_collided -= collided
            if not newly_collided:
                break
            collided.update(newly_collided)
            for index in newly_collided:
                resolved[index] = old_positions[index]

        return tuple(resolved), frozenset(collided)

    def _is_transition_collision(
        self, old_positions: Tuple[Point, ...], new_positions: Tuple[Point, ...]
    ) -> bool:
        if len(set(new_positions)) != len(new_positions):
            return True
        for left in range(len(old_positions)):
            for right in range(left + 1, len(old_positions)):
                if (
                    new_positions[left] == old_positions[right]
                    and new_positions[right] == old_positions[left]
                ):
                    return True
                if self.config.strict_agent_obstacles:
                    left_entered_occupied = (
                        new_positions[left] != old_positions[left]
                        and new_positions[left] == old_positions[right]
                    )
                    right_entered_occupied = (
                        new_positions[right] != old_positions[right]
                        and new_positions[right] == old_positions[left]
                    )
                    if left_entered_occupied or right_entered_occupied:
                        return True
        return False

    def _step_environment_effects(
        self, state: BurgerState, events: List[Dict[str, object]]
    ) -> None:
        state.timestep += 1
        grill = state.grill
        if grill.food == "raw_beef":
            grill.cook_ticks += 1
            if grill.cook_ticks == self.config.cook_steps:
                grill.food = "cooked_beef"
                grill.ready_ticks = 0
                events.append({"type": "beef_ready"})
        elif grill.food == "cooked_beef":
            grill.ready_ticks += 1
            if grill.ready_ticks == self.config.burn_steps:
                grill.food = "burnt_beef"
                events.append({"type": "fire_started"})

        returned = [
            due for due in state.pending_plate_returns if due <= state.timestep
        ]
        if returned:
            state.dirty_plates_at_return += len(returned)
            state.pending_plate_returns = [
                due
                for due in state.pending_plate_returns
                if due > state.timestep
            ]
            events.append({"type": "dirty_plates_returned", "count": len(returned)})

    def potential(self, state: BurgerState) -> float:
        """Bounded recipe-progress potential; never a repeated state bonus."""

        world_items: List[Item] = [
            player.held_object
            for player in state.players
            if player.held_object is not None
        ]
        world_items.extend(state.counter_objects.values())
        if state.grill.food is not None:
            world_items.append(state.grill.food)

        plated_contents = set().union(
            *(
                PLATE_CONTENTS[item]
                for item in world_items
                if item in PLATE_CONTENTS
            ),
            set(),
        )
        has_bun = "bun" in world_items or "bun" in plated_contents
        has_lettuce = "lettuce" in world_items or "lettuce" in plated_contents
        has_raw = "raw_beef" in world_items
        has_cooked = (
            "cooked_beef" in world_items
            or "cooked_beef" in plated_contents
        )
        # A dirty plate is conserved physical inventory, but it cannot receive
        # ingredients and must not count as recipe assembly progress.
        has_plate = any(item in PLATE_CONTENTS for item in world_items)
        has_burger = "plated_burger" in world_items
        grill_raw = state.grill.food == "raw_beef"
        cooking_started = (
            state.grill.food in {"raw_beef", "cooked_beef"}
            or has_cooked
            or has_burger
        )
        grill_progress = (
            state.grill.cook_ticks / self.config.cook_steps
            if grill_raw
            else 0.0
        )
        recipe_progress = float(
            cooking_started * has_bun
            + cooking_started * has_lettuce
            + 0.5 * has_raw
            + 0.5 * grill_raw
            + 0.5 * grill_progress
            + 1.5 * has_cooked
            + cooking_started * has_plate
            + 2 * has_burger
        )
        partial_plate_items = [
            item
            for item in world_items
            if item in PLATE_CONTENTS
            and 0 < len(PLATE_CONTENTS[item]) < 3
        ]
        partial_plate_parked = any(
            item in PLATE_CONTENTS
            and 0 < len(PLATE_CONTENTS[item]) < 3
            for item in state.counter_objects.values()
        )
        beef_workflow_active = bool(
            has_raw
            or state.grill.food in {"raw_beef", "cooked_beef"}
            or has_cooked
        )
        # Parking an incomplete plate is a required, conserved state change
        # when the hand must be freed to fetch raw beef. Keep that workflow
        # phase active while beef is being prepared so the correct drop does
        # not create a local potential valley. Because this is part of Phi(s),
        # dropping and immediately picking the plate back still telescopes to
        # a net loss and cannot be farmed for reward.
        assembly_workflow_progress = (
            0.75
            if partial_plate_parked
            or (partial_plate_items and beef_workflow_active)
            else 0.0
        )
        dish_cycle_progress = 0.0
        if state.clean_plates == 0:
            held_items = {
                player.held_object
                for player in state.players
                if player.held_object is not None
            }
            if "dirty_plate" in held_items:
                dish_cycle_progress = 0.5
            elif state.sink.has_dirty_plate:
                dish_cycle_progress = (
                    0.5
                    + 0.5
                    * state.sink.wash_progress
                    / self.config.wash_steps
                )
            elif any(item in PLATE_CONTENTS for item in world_items):
                dish_cycle_progress = 1.0
        fire_recovery_potential = 0.0
        if state.grill.food == "burnt_beef":
            held_items = [
                player.held_object for player in state.players
            ]
            if "extinguisher" in held_items:
                fire_recovery_potential = -6.0
            elif None in held_items:
                fire_recovery_potential = -9.0
            else:
                # Freeing the hand is the largest recovery bottleneck. Make
                # immediately taking the parked item back a clear regression.
                fire_recovery_potential = -18.0
        elif any(
            player.held_object == "extinguisher"
            for player in state.players
        ):
            # Clearing the grill is not the end of recovery: the conserved
            # extinguisher must be returned before cooking can resume.
            fire_recovery_potential = -3.0
        return (
            POTENTIAL_FLOOR
            + recipe_progress
            + assembly_workflow_progress
            + dish_cycle_progress
            + fire_recovery_potential
            + self.config.reward.navigation_potential_scale
            * self._single_agent_navigation_potential(state)
        )

    def _single_agent_navigation_potential(
        self, state: BurgerState
    ) -> float:
        """Dense path progress for PPO, expressed only as state potential."""

        if len(state.players) != 1:
            return 0.0
        player = state.players[0]
        held = player.held_object
        target: Optional[str] = None
        if state.grill.food == "burnt_beef":
            if held == "extinguisher":
                target = GRILL
            elif held is None:
                target = EXTINGUISHER
            else:
                # A hand must be freed before the extinguisher can be taken.
                return self._counter_navigation_potential(
                    state, player.position, require_empty=True
                )
        elif held == "extinguisher":
            target = EXTINGUISHER
        elif held == "plated_burger":
            target = SERVE
        elif held == "dirty_plate":
            target = SINK
        elif held == "raw_beef" and state.grill.food is None:
            target = GRILL
        elif held in PLATE_CONTENTS:
            contents = PLATE_CONTENTS[held]
            if state.grill.food is None:
                if held == "clean_plate":
                    target = PLATE_RACK
                else:
                    # A partial plate cannot take raw beef directly. Park it
                    # on an empty worktop before fetching beef for the grill.
                    return self._counter_navigation_potential(
                        state, player.position, require_empty=True
                    )
            elif "bun" not in contents:
                target = BUN_DISPENSER
            elif "lettuce" not in contents:
                target = LETTUCE_DISPENSER
            elif "cooked_beef" not in contents:
                target = GRILL
        elif held is None:
            if (
                state.clean_plates == 0
                and state.sink.has_dirty_plate
            ):
                target = SINK
            elif (
                state.clean_plates == 0
                and (
                    state.dirty_plates_at_return > 0
                    or state.pending_plate_returns
                )
            ):
                target = RETURN
            elif state.grill.food is None:
                target = BEEF_DISPENSER
            elif state.clean_plates == 0 and any(
                item in PLATE_CONTENTS
                for item in state.counter_objects.values()
            ):
                # A washed plate may have been parked on a worktop during
                # fire recovery. The empty rack is not a valid objective.
                return self._counter_navigation_potential(
                    state,
                    player.position,
                    require_plate=True,
                )
            else:
                target = PLATE_RACK
        if target is None:
            return 0.0
        distance = self._station_distances.get(target, {}).get(
            player.position
        )
        if distance is None:
            return 0.0
        normalizer = max(self.layout.width + self.layout.height - 2, 1)
        return max(0.0, 1.0 - distance / normalizer)

    def _counter_navigation_potential(
        self,
        state: BurgerState,
        position: Point,
        *,
        require_empty: bool = False,
        require_plate: bool = False,
    ) -> float:
        """Shortest-path potential to a context-valid worktop objective."""

        distances = []
        for counter, distance_field in self._counter_distances.items():
            item = state.counter_objects.get(counter)
            if require_empty and item is not None:
                continue
            if require_plate and item not in PLATE_CONTENTS:
                continue
            distance = distance_field.get(position)
            if distance is not None:
                distances.append(distance)
        if not distances:
            return 0.0
        normalizer = max(self.layout.width + self.layout.height - 2, 1)
        return max(0.0, 1.0 - min(distances) / normalizer)

    def _reward_breakdown(
        self,
        events: Sequence[Mapping[str, object]],
        old_phi: float,
        new_phi: float,
    ) -> Dict[str, float]:
        counts: Dict[str, int] = {}
        for event in events:
            event_type = str(event["type"])
            counts[event_type] = counts.get(event_type, 0) + 1
        reward = self.config.reward
        return {
            "correct_delivery": counts.get("correct_delivery", 0)
            * reward.correct_delivery,
            "raw_beef_placed": counts.get("raw_beef_placed", 0)
            * reward.raw_beef_placed,
            "dirty_plate_pickup": counts.get("dirty_plate_pickup", 0)
            * reward.dirty_plate_pickup,
            "wash_started": counts.get("wash_started", 0)
            * reward.wash_started,
            "wash_progress": counts.get("wash_progress", 0)
            * reward.wash_progress,
            "plate_washed": counts.get("plate_washed", 0)
            * reward.plate_washed,
            "fire_extinguished": counts.get("fire_extinguished", 0)
            * reward.fire_extinguished,
            "fire_started": counts.get("fire_started", 0) * reward.fire_started,
            "fire_active": counts.get("fire_active", 0) * reward.fire_active,
            "fire_food_handling": counts.get("fire_food_handling", 0)
            * reward.fire_food_handling,
            "dirty_plate_counter_handling": counts.get(
                "dirty_plate_counter_handling", 0
            )
            * reward.dirty_plate_counter_handling,
            "collision": counts.get("collision", 0) * reward.collision,
            "time_step": reward.time_step,
            "potential": reward.potential_scale
            * (reward.gamma * new_phi - old_phi),
        }

    def validate_state(self, state: BurgerState) -> None:
        if not 1 <= len(state.players) <= 4:
            raise ValueError("State must contain one to four players")
        positions = [player.position for player in state.players]
        if len(set(positions)) != len(positions):
            raise ValueError("Players cannot share a grid square")
        if any(pos not in self.layout.valid_player_positions for pos in positions):
            raise ValueError("Player is on non-walkable terrain")
        for player in state.players:
            if player.orientation not in Direction.ALL_DIRECTIONS:
                raise ValueError("Invalid player orientation")
            self._validate_item(player.held_object)
            if player.held_object == "cooked_beef":
                raise ValueError("Cooked beef cannot be held without a plate")
        for position, item in state.counter_objects.items():
            if self.layout.terrain_at(position) != COUNTER:
                raise ValueError("Counter object is not located on a counter")
            self._validate_item(item)

        integer_counts = {
            "timestep": state.timestep,
            "clean_plates": state.clean_plates,
            "dirty_plates_at_return": state.dirty_plates_at_return,
            "total_plates": state.total_plates,
            "delivered_orders": state.delivered_orders,
            "grill.cook_ticks": state.grill.cook_ticks,
            "grill.ready_ticks": state.grill.ready_ticks,
            "sink.wash_progress": state.sink.wash_progress,
        }
        if any(
            not isinstance(value, int) or value < 0
            for value in integer_counts.values()
        ):
            raise ValueError("State counters must be non-negative integers")
        if not isinstance(state.extinguisher_available, bool):
            raise ValueError("Extinguisher availability must be boolean")
        if any(
            not isinstance(due, int) or due < state.timestep
            for due in state.pending_plate_returns
        ):
            raise ValueError("Pending plate return has an invalid due timestep")

        if state.grill.food not in {
            None,
            "raw_beef",
            "cooked_beef",
            "burnt_beef",
        }:
            raise ValueError("Invalid grill food")
        if state.grill.food is None and (
            state.grill.cook_ticks != 0 or state.grill.ready_ticks != 0
        ):
            raise ValueError("Empty grill cannot retain timer progress")
        if state.grill.cook_ticks > self.config.cook_steps:
            raise ValueError("Grill cook timer exceeded configured duration")
        if state.grill.ready_ticks > self.config.burn_steps:
            raise ValueError("Grill burn timer exceeded configured duration")

        if (
            not state.sink.has_dirty_plate
            and state.sink.wash_progress != 0
        ):
            raise ValueError("Empty sink cannot retain washing state")
        if state.sink.has_dirty_plate:
            if state.sink.wash_progress >= self.config.wash_steps:
                raise ValueError("Completed wash must immediately return a clean plate")

        observed_plates = (
            state.clean_plates
            + state.dirty_plates_at_return
            + len(state.pending_plate_returns)
            + int(state.sink.has_dirty_plate)
            + sum(
                player.held_object in PLATE_ITEMS for player in state.players
            )
            + sum(item in PLATE_ITEMS for item in state.counter_objects.values())
        )
        if observed_plates != state.total_plates:
            raise ValueError(
                "Plate conservation violated: observed {}, expected {}".format(
                    observed_plates, state.total_plates
                )
            )

        observed_extinguishers = (
            int(state.extinguisher_available)
            + sum(
                player.held_object == "extinguisher"
                for player in state.players
            )
            + sum(
                item == "extinguisher"
                for item in state.counter_objects.values()
            )
        )
        if observed_extinguishers != 1:
            raise ValueError(
                "Extinguisher conservation violated: observed {}, expected 1".format(
                    observed_extinguishers
                )
            )

    @staticmethod
    def _validate_item(item: Optional[Item]) -> None:
        if item is not None and item not in VALID_ITEMS:
            raise ValueError("Invalid burger item: {!r}".format(item))

    @staticmethod
    def _combine_plate_and_ingredient(
        plate: Optional[Item], ingredient: Optional[Item]
    ) -> Optional[Item]:
        if plate not in PLATE_CONTENTS:
            return None
        if ingredient not in {"bun", "lettuce", "cooked_beef"}:
            return None
        contents = PLATE_CONTENTS[plate]
        if ingredient in contents:
            return None
        return CONTENTS_TO_PLATE[contents | {ingredient}]

    def state_digest(self, state: BurgerState) -> str:
        """Stable digest used to detect replay/state tampering."""

        payload = {
            "players": [
                {
                    "position": player.position,
                    "orientation": player.orientation,
                    "held_object": player.held_object,
                }
                for player in state.players
            ],
            "counter_objects": sorted(
                (position, item) for position, item in state.counter_objects.items()
            ),
            "timestep": state.timestep,
            "clean_plates": state.clean_plates,
            "dirty_plates_at_return": state.dirty_plates_at_return,
            "pending_plate_returns": sorted(state.pending_plate_returns),
            "total_plates": state.total_plates,
            "grill": vars(state.grill),
            "sink": vars(state.sink),
            "delivered_orders": state.delivered_orders,
            "extinguisher_available": state.extinguisher_available,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class BurgerEnv:
    """State-owning facade suitable for a Gym/MAPPO adapter."""

    def __init__(
        self,
        mdp: Optional[BurgerGridworld] = None,
        num_players: int = 4,
        start_stage: str = "standard",
        randomize_player_positions: bool = False,
        random_start_max_objective_distance: Optional[int] = None,
        random_start_candidate_positions: Sequence[Point] = (),
    ) -> None:
        if (
            random_start_max_objective_distance is not None
            and random_start_max_objective_distance < 0
        ):
            raise ValueError(
                "random_start_max_objective_distance must be non-negative"
            )
        if (
            random_start_max_objective_distance is not None
            and not randomize_player_positions
        ):
            raise ValueError(
                "random_start_max_objective_distance requires randomized "
                "player positions"
            )
        if random_start_candidate_positions and not randomize_player_positions:
            raise ValueError(
                "random_start_candidate_positions requires randomized "
                "player positions"
            )
        self.mdp = mdp or BurgerGridworld()
        self.num_players = num_players
        self.start_stage = start_stage
        self.randomize_player_positions = randomize_player_positions
        self.random_start_max_objective_distance = (
            random_start_max_objective_distance
        )
        self.random_start_candidate_positions = tuple(
            random_start_candidate_positions
        )
        invalid_candidates = set(
            self.random_start_candidate_positions
        ) - set(self.mdp.layout.valid_player_positions)
        if invalid_candidates:
            raise ValueError(
                "Random start candidates must be walkable floor cells: "
                "{}".format(sorted(invalid_candidates))
            )
        self._rng = random.Random(0)
        self._state = self._new_start_state()

    @property
    def state(self) -> BurgerState:
        return copy.deepcopy(self._state)

    @property
    def _state_view(self) -> BurgerState:
        """Internal read-only state view for the high-throughput adapter."""

        return self._state

    def _new_start_state(self) -> BurgerState:
        state = self.mdp.get_start_state(
            self.num_players, stage=self.start_stage
        )
        if self.randomize_player_positions:
            candidate_positions = sorted(
                self.random_start_candidate_positions
                or self.mdp.layout.valid_player_positions
            )
            target = self._start_objective_target(state)
            if (
                target is not None
                and self.random_start_max_objective_distance is not None
            ):
                distance_field = self.mdp._station_distances[target]
                candidate_positions = [
                    position
                    for position in candidate_positions
                    if distance_field.get(position)
                    is not None
                    and distance_field[position]
                    <= self.random_start_max_objective_distance
                ]
            if len(candidate_positions) < self.num_players:
                raise ValueError(
                    "Distance-limited randomized start has fewer legal "
                    "positions than players"
                )
            positions = self._rng.sample(
                candidate_positions,
                self.num_players,
            )
            for player, position in zip(state.players, positions):
                player.position = position
            self.mdp.validate_state(state)
        return state

    @staticmethod
    def _start_objective_target(state: BurgerState) -> Optional[str]:
        """Return the recovery station for hard-start distance curricula."""

        held_items = {
            player.held_object
            for player in state.players
            if player.held_object is not None
        }
        if state.grill.food == "burnt_beef":
            if "extinguisher" in held_items:
                return GRILL
            if any(
                player.held_object is None for player in state.players
            ):
                return EXTINGUISHER
            return None
        if "dirty_plate" in held_items:
            return SINK
        if state.clean_plates == 0 and state.sink.has_dirty_plate:
            return SINK
        if (
            state.clean_plates == 0
            and (
                state.dirty_plates_at_return > 0
                or state.pending_plate_returns
            )
        ):
            return RETURN
        return None

    def reset(self, seed: Optional[int] = None) -> BurgerState:
        if seed is not None:
            if not isinstance(seed, int):
                raise ValueError("seed must be an integer")
            self._rng.seed(seed)
        self._state = self._new_start_state()
        return self.state

    def step(self, joint_action: Sequence[object]) -> BurgerTransition:
        transition = self._step_internal(joint_action)
        return BurgerTransition(
            state=copy.deepcopy(transition.state),
            reward=transition.reward,
            done=transition.done,
            info=copy.deepcopy(transition.info),
        )

    def _step_internal(
        self, joint_action: Sequence[object]
    ) -> BurgerTransition:
        """Advance state without redundant defensive output copies.

        This is reserved for the MAPPO adapter, which immediately encodes the
        result and never exposes the mutable state object to a policy.
        """

        transition = self.mdp.get_state_transition(self._state, joint_action)
        self._state = transition.state
        return transition


__all__ = [
    "BurgerConfig",
    "BurgerEnv",
    "BurgerGridworld",
    "BurgerLayout",
    "BurgerPlayerState",
    "BurgerRewardConfig",
    "BurgerState",
    "BurgerTransition",
    "CRAMPED_GALLEY",
    "CURRICULUM_START_STAGES",
    "RETURN",
    "TRASH",
]
