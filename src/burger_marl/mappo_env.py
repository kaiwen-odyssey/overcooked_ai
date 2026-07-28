"""Fixed-shape CTDE adapter for PPO and multi-agent PPO.

Actors receive only world-aligned local semantic observations. The centralized
critic receives a separate full-state tensor. The adapter owns the underlying
environment state, accepts only seven-action indices, and repeats the validated
team reward for active agents.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium
import numpy as np

from burger_marl.actions import Action
from burger_marl.env import (
    BEEF_DISPENSER,
    BUN_DISPENSER,
    CONTENTS_TO_PLATE,
    COUNTER,
    EXTINGUISHER,
    FLOOR,
    GRILL,
    LETTUCE_DISPENSER,
    PLATE_CONTENTS,
    PLATE_RACK,
    RETURN,
    SERVE,
    SINK,
    TRASH,
    BurgerEnv,
    BurgerGridworld,
    BurgerState,
)

Point = Tuple[int, int]

TERRAIN_ORDER = (
    None,
    FLOOR,
    COUNTER,
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
)
TERRAIN_CHANNEL = {
    terrain: index for index, terrain in enumerate(TERRAIN_ORDER)
}
_LOCAL_SEMANTIC_OFFSET = len(TERRAIN_ORDER)
LOCAL_CHANNELS = {
    "self": _LOCAL_SEMANTIC_OFFSET,
    "teammate": _LOCAL_SEMANTIC_OFFSET + 1,
    "bun": _LOCAL_SEMANTIC_OFFSET + 2,
    "lettuce": _LOCAL_SEMANTIC_OFFSET + 3,
    "raw_beef": _LOCAL_SEMANTIC_OFFSET + 4,
    "cooked_beef": _LOCAL_SEMANTIC_OFFSET + 5,
    "burnt_beef": _LOCAL_SEMANTIC_OFFSET + 6,
    "plate": _LOCAL_SEMANTIC_OFFSET + 7,
    "plated_burger": _LOCAL_SEMANTIC_OFFSET + 8,
    "extinguisher": _LOCAL_SEMANTIC_OFFSET + 9,
    "station_progress": _LOCAL_SEMANTIC_OFFSET + 10,
    "stack_fill": _LOCAL_SEMANTIC_OFFSET + 11,
    "time_remaining": _LOCAL_SEMANTIC_OFFSET + 12,
}
NUM_LOCAL_CHANNELS = max(LOCAL_CHANNELS.values()) + 1
MAX_AGENTS = 4
GLOBAL_AGENT_OFFSET = len(TERRAIN_ORDER)
GLOBAL_ORIENTATION_OFFSET = GLOBAL_AGENT_OFFSET + MAX_AGENTS
GLOBAL_ITEM_OFFSET = GLOBAL_ORIENTATION_OFFSET + MAX_AGENTS * 4
GLOBAL_CHANNELS = {
    "bun": GLOBAL_ITEM_OFFSET,
    "lettuce": GLOBAL_ITEM_OFFSET + 1,
    "raw_beef": GLOBAL_ITEM_OFFSET + 2,
    "cooked_beef": GLOBAL_ITEM_OFFSET + 3,
    "burnt_beef": GLOBAL_ITEM_OFFSET + 4,
    "plate": GLOBAL_ITEM_OFFSET + 5,
    "plated_burger": GLOBAL_ITEM_OFFSET + 6,
    "extinguisher": GLOBAL_ITEM_OFFSET + 7,
    "station_progress": GLOBAL_ITEM_OFFSET + 8,
    "stack_fill": GLOBAL_ITEM_OFFSET + 9,
    "time_remaining": GLOBAL_ITEM_OFFSET + 10,
}
NUM_GLOBAL_CHANNELS = GLOBAL_ITEM_OFFSET + 11
NUM_GLOBAL_SCALARS = MAX_AGENTS * 2
ORIENTATION_TO_INDEX = {
    (0, -1): 0,
    (0, 1): 1,
    (1, 0): 2,
    (-1, 0): 3,
}


class BurgerMAPPOEnv:
    """Framework-neutral MAPPO contract with four fixed agent slots."""

    def __init__(
        self,
        mdp: Optional[BurgerGridworld] = None,
        num_players: int = 4,
        observation_radius: int = 4,
        occlusion: bool = True,
    ) -> None:
        if not 1 <= num_players <= MAX_AGENTS:
            raise ValueError("num_players must be between one and four")
        if observation_radius < 1:
            raise ValueError("observation_radius must be positive")
        self.mdp = mdp or BurgerGridworld()
        self.num_players = num_players
        self.num_agents = MAX_AGENTS
        self.observation_radius = observation_radius
        self.occlusion = occlusion
        self._side = observation_radius * 2 + 1
        self._env = BurgerEnv(self.mdp, num_players)

        local_space = gymnasium.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(NUM_LOCAL_CHANNELS, self._side, self._side),
            dtype=np.float32,
        )
        global_dim = (
            NUM_GLOBAL_CHANNELS
            * self.mdp.layout.height
            * self.mdp.layout.width
            + NUM_GLOBAL_SCALARS
        )
        shared_space = gymnasium.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(global_dim,),
            dtype=np.float32,
        )
        self.observation_space = tuple(
            copy.deepcopy(local_space) for _ in range(MAX_AGENTS)
        )
        self.share_observation_space = tuple(
            copy.deepcopy(shared_space) for _ in range(MAX_AGENTS)
        )
        self.action_space = tuple(
            gymnasium.spaces.Discrete(Action.NUM_ACTIONS)
            for _ in range(MAX_AGENTS)
        )

    @property
    def state(self) -> BurgerState:
        return self._env.state

    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Dynamics are deterministic. Accepting a seed keeps vectorized trainer
        # interfaces stable without introducing hidden RNG state.
        if seed is not None and not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        self._env.reset()
        return (
            self.local_observations(),
            self.shared_observations(),
            self.available_actions(),
        )

    def step(
        self, actions: Sequence[int]
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[Dict[str, object]],
        np.ndarray,
    ]:
        if len(actions) != MAX_AGENTS:
            raise ValueError("MAPPO adapter requires four fixed action slots")
        action_indices: List[int] = []
        for index, action in enumerate(actions):
            if not isinstance(action, (int, np.integer)):
                raise ValueError("Action indices must be integers")
            action_index = int(action)
            if not self.action_space[index].contains(action_index):
                raise ValueError("Action index out of range")
            action_indices.append(action_index)

        stay_index = Action.ACTION_TO_INDEX[Action.STAY]
        if any(
            action_indices[index] != stay_index
            for index in range(self.num_players, MAX_AGENTS)
        ):
            raise ValueError("Inactive MAPPO slots must submit STAY")

        joint_action = [
            Action.INDEX_TO_ACTION[index]
            for index in action_indices[: self.num_players]
        ]
        transition = self._env.step(joint_action)
        observations = self.local_observations()
        shared_observations = self.shared_observations()
        rewards = np.zeros((MAX_AGENTS, 1), dtype=np.float32)
        rewards[: self.num_players, 0] = transition.reward
        dones = np.ones(MAX_AGENTS, dtype=np.bool_)
        dones[: self.num_players] = transition.done
        infos: List[Dict[str, object]] = []
        for index in range(MAX_AGENTS):
            active = index < self.num_players
            info = {
                "active": active,
                "active_mask": float(active),
                "authoritative_state": True,
            }
            if active:
                info.update(copy.deepcopy(dict(transition.info)))
            infos.append(info)
        return (
            observations,
            shared_observations,
            rewards,
            dones,
            infos,
            self.available_actions(),
        )

    def active_masks(self) -> np.ndarray:
        masks = np.zeros((MAX_AGENTS, 1), dtype=np.float32)
        masks[: self.num_players] = 1.0
        return masks

    def available_actions(self) -> np.ndarray:
        """Return local-visible masks for world-aligned movement and work.

        Static terrain-blocked moves are masked because facing is not part of
        the v2 interaction contract. Moves toward a teammate remain available:
        lockstep resolution decides whether that teammate vacates the square.
        Inactive fixed slots can only submit STAY.
        """

        available = np.zeros(
            (MAX_AGENTS, Action.NUM_ACTIONS), dtype=np.float32
        )
        for player_idx in range(self.num_players):
            player = self._env.state.players[player_idx]
            available[
                player_idx,
                Action.ACTION_TO_INDEX[Action.STAY],
            ] = 1.0
            for action in Action.MOTION_ACTIONS:
                if action == Action.STAY:
                    continue
                target = Action.move_in_direction(player.position, action)
                if target in self.mdp.layout.valid_player_positions:
                    available[
                        player_idx,
                        Action.ACTION_TO_INDEX[action],
                    ] = 1.0
            for context_action in (Action.PICK_DROP, Action.PROCESS):
                if self.mdp.context_action_available(
                    self._env.state,
                    player_idx,
                    context_action,
                ):
                    available[
                        player_idx,
                        Action.ACTION_TO_INDEX[context_action],
                    ] = 1.0
        available[
            self.num_players :,
            Action.ACTION_TO_INDEX[Action.STAY],
        ] = 1.0
        active = available[: self.num_players]
        stay_index = Action.ACTION_TO_INDEX[Action.STAY]
        if np.any(active.sum(axis=-1) < 1) or np.any(
            active[:, stay_index] != 1
        ):
            raise RuntimeError(
                "Active action mask froze or removed the STAY fallback"
            )
        return available

    def local_observations(self) -> np.ndarray:
        observations = np.zeros(
            (
                MAX_AGENTS,
                NUM_LOCAL_CHANNELS,
                self._side,
                self._side,
            ),
            dtype=np.float32,
        )
        state = self._env.state
        for agent_index in range(self.num_players):
            observations[agent_index] = self._local_observation(
                state, agent_index
            )
        return observations

    def shared_observations(self) -> np.ndarray:
        shared = np.zeros(
            (
                MAX_AGENTS,
                self.share_observation_space[0].shape[0],
            ),
            dtype=np.float32,
        )
        encoded = self._global_state(self._env.state)
        shared[: self.num_players] = encoded
        return shared

    def _local_observation(
        self, state: BurgerState, agent_index: int
    ) -> np.ndarray:
        observation = np.zeros(
            (NUM_LOCAL_CHANNELS, self._side, self._side),
            dtype=np.float32,
        )
        player = state.players[agent_index]
        center = self.observation_radius
        player_by_position = {
            other.position: index
            for index, other in enumerate(state.players)
        }
        for local_y in range(self._side):
            for local_x in range(self._side):
                dx = local_x - center
                dy = local_y - center
                world = (
                    player.position[0] + dx,
                    player.position[1] + dy,
                )
                visible = (
                    not self.occlusion
                    or self._has_line_of_sight(player.position, world)
                )
                terrain = (
                    self.mdp.layout.terrain_at(world)
                    if visible
                    else None
                )
                observation[
                    TERRAIN_CHANNEL[terrain], local_y, local_x
                ] = 1.0
                if not visible or terrain is None:
                    continue
                channel_values = observation[:, local_y, local_x]
                occupying_agent = player_by_position.get(world)
                if occupying_agent is not None:
                    channel_values[
                        LOCAL_CHANNELS[
                            "self"
                            if occupying_agent == agent_index
                            else "teammate"
                        ]
                    ] = 1.0
                    self._encode_item(
                        channel_values,
                        state.players[occupying_agent].held_object,
                        LOCAL_CHANNELS,
                    )
                self._encode_world_cell(
                    state, world, channel_values, LOCAL_CHANNELS
                )
        observation[LOCAL_CHANNELS["time_remaining"]] = max(
            0.0,
            1.0 - state.timestep / self.mdp.config.horizon,
        )
        return observation

    def _global_state(self, state: BurgerState) -> np.ndarray:
        encoded = np.zeros(
            (
                NUM_GLOBAL_CHANNELS,
                self.mdp.layout.height,
                self.mdp.layout.width,
            ),
            dtype=np.float32,
        )
        for y in range(self.mdp.layout.height):
            for x in range(self.mdp.layout.width):
                world = (x, y)
                terrain = self.mdp.layout.terrain_at(world)
                encoded[TERRAIN_CHANNEL[terrain], y, x] = 1.0
                values = encoded[:, y, x]
                self._encode_world_cell(
                    state, world, values, GLOBAL_CHANNELS
                )
        for agent_index, player in enumerate(state.players):
            x, y = player.position
            encoded[GLOBAL_AGENT_OFFSET + agent_index, y, x] = 1.0
            orientation_index = ORIENTATION_TO_INDEX[player.orientation]
            orientation_channel = (
                GLOBAL_ORIENTATION_OFFSET
                + agent_index * 4
                + orientation_index
            )
            encoded[orientation_channel, y, x] = 1.0
            self._encode_item(
                encoded[:, y, x],
                player.held_object,
                GLOBAL_CHANNELS,
            )
        encoded[GLOBAL_CHANNELS["time_remaining"]] = max(
            0.0,
            1.0 - state.timestep / self.mdp.config.horizon,
        )
        active = np.zeros(MAX_AGENTS, dtype=np.float32)
        active[: self.num_players] = 1.0
        pending_returns = np.zeros(MAX_AGENTS, dtype=np.float32)
        for index, due in enumerate(sorted(state.pending_plate_returns)):
            pending_returns[index] = min(
                1.0,
                (due - state.timestep)
                / self.mdp.config.plate_return_steps,
            )
        return np.concatenate(
            (
                encoded.reshape(-1),
                active,
                pending_returns,
            )
        )

    def _encode_world_cell(
        self,
        state: BurgerState,
        world: Point,
        values: np.ndarray,
        channels: Dict[str, int],
    ) -> None:
        self._encode_item(
            values, state.counter_objects.get(world), channels
        )
        terrain = self.mdp.layout.terrain_at(world)
        if terrain == GRILL:
            self._encode_item(values, state.grill.food, channels)
            if state.grill.food == "raw_beef":
                values[channels["station_progress"]] = (
                    state.grill.cook_ticks / self.mdp.config.cook_steps
                )
            elif state.grill.food == "cooked_beef":
                values[channels["station_progress"]] = (
                    state.grill.ready_ticks / self.mdp.config.burn_steps
                )
            elif state.grill.food == "burnt_beef":
                values[channels["station_progress"]] = 1.0
        elif terrain == SINK and state.sink.has_dirty_plate:
            self._encode_item(values, "dirty_plate", channels)
            values[channels["station_progress"]] = (
                state.sink.wash_progress / self.mdp.config.wash_steps
            )
        elif terrain == RETURN:
            values[channels["stack_fill"]] = min(
                1.0,
                state.dirty_plates_at_return / state.total_plates,
            )
        elif terrain == PLATE_RACK:
            if state.clean_plates:
                self._encode_item(values, "clean_plate", channels)
            values[channels["stack_fill"]] = min(
                1.0, state.clean_plates / state.total_plates
            )
        elif terrain == EXTINGUISHER and state.extinguisher_available:
            self._encode_item(values, "extinguisher", channels)

    @staticmethod
    def _encode_item(
        values: np.ndarray,
        item: Optional[str],
        channels: Dict[str, int],
    ) -> None:
        if item is None:
            return
        if item in PLATE_CONTENTS:
            values[channels["plate"]] = 1.0
            for ingredient in PLATE_CONTENTS[item]:
                values[channels[ingredient]] = 1.0
            if item == CONTENTS_TO_PLATE[
                frozenset({"bun", "lettuce", "cooked_beef"})
            ]:
                values[channels["plated_burger"]] = 1.0
            return
        if item == "dirty_plate":
            values[channels["plate"]] = 1.0
        elif item in channels:
            values[channels[item]] = 1.0

    def _has_line_of_sight(self, origin: Point, target: Point) -> bool:
        if self.mdp.layout.terrain_at(target) is None:
            return True
        line = self._bresenham(origin, target)
        for point in line[1:-1]:
            if self.mdp.layout.terrain_at(point) != FLOOR:
                return False
        return True

    @staticmethod
    def _bresenham(start: Point, end: Point) -> List[Point]:
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        step_x = 1 if x0 < x1 else -1
        step_y = 1 if y0 < y1 else -1
        error = dx + dy
        points = []
        while True:
            points.append((x0, y0))
            if (x0, y0) == (x1, y1):
                break
            doubled = 2 * error
            if doubled >= dy:
                error += dy
                x0 += step_x
            if doubled <= dx:
                error += dx
                y0 += step_y
        return points


__all__ = [
    "BurgerMAPPOEnv",
    "GLOBAL_CHANNELS",
    "GLOBAL_ORIENTATION_OFFSET",
    "LOCAL_CHANNELS",
    "MAX_AGENTS",
    "NUM_GLOBAL_CHANNELS",
    "NUM_LOCAL_CHANNELS",
]
