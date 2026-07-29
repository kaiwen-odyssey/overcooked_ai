import copy
import random
import unittest
from collections import deque
from itertools import permutations, product

from burger_marl.actions import Action, Direction
from burger_marl.env import (
    BurgerConfig,
    BurgerEnv,
    BurgerGridworld,
    BurgerLayout,
    BurgerPlayerState,
    BurgerRewardConfig,
    BurgerState,
    GrillState,
    PLATE_CONTENTS,
    SinkState,
)


def layout(rows, starts):
    """Add unreachable required stations so focused layouts remain valid."""

    width = max(11, *(len(row) for row in rows))
    return BurgerLayout(
        rows=tuple(row.ljust(width) for row in rows)
        + ("BLMGDWSETR".ljust(width),),
        player_starts=tuple(starts),
        name="focused_test",
    )


class TestBurgerGridworldContract(unittest.TestCase):
    def _move_next_to(self, env, station):
        terrain = env.mdp.layout
        start = env.state.players[0].position
        station_x, station_y = station
        targets = {
            (station_x + dx, station_y + dy)
            for dx, dy in Direction.ALL_DIRECTIONS
            if terrain.terrain_at((station_x + dx, station_y + dy)) == " "
        }
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            position, path = queue.popleft()
            if position in targets:
                for action in path:
                    env.step([action])
                return
            for direction in Direction.ALL_DIRECTIONS:
                next_position = Action.move_in_direction(position, direction)
                if (
                    next_position not in visited
                    and terrain.terrain_at(next_position) == " "
                ):
                    visited.add(next_position)
                    queue.append((next_position, path + [direction]))
        self.fail("Station at {} is unreachable".format(station))

    def _interact_with(self, env, station):
        self._move_next_to(env, station)
        return env.step([Action.PICK_DROP])

    def test_default_map_single_agent_can_complete_full_burger_loop(self):
        env = BurgerEnv(BurgerGridworld(), num_players=1)
        stations = {
            "bun": (0, 0),
            "lettuce": (1, 0),
            "beef": (3, 0),
            "grill": (5, 0),
            "plate": (1, 7),
            "serve": (7, 7),
        }

        self._interact_with(env, stations["beef"])
        self._interact_with(env, stations["grill"])
        self._interact_with(env, stations["plate"])
        self._interact_with(env, stations["bun"])
        self._interact_with(env, stations["lettuce"])
        self._move_next_to(env, stations["grill"])
        while env.state.grill.food == "raw_beef":
            env.step([Action.STAY])
        self._interact_with(env, stations["grill"])
        delivery = self._interact_with(env, stations["serve"])

        self.assertEqual(delivery.state.delivered_orders, 1)
        self.assertEqual(
            delivery.info["reward_breakdown"]["correct_delivery"], 20.0
        )
        env.mdp.validate_state(delivery.state)

    def test_direct_ppo_curriculum_starts_are_legal_physical_states(self):
        mdp = BurgerGridworld()
        expected = {
            "serve_ready": ("plated_burger", None),
            "grill_ready": ("plate_bun_lettuce", "cooked_beef"),
            "cook_ready": ("plate_bun_lettuce", "raw_beef"),
            "plate_ready": ("clean_plate", "raw_beef"),
            "plate_pick_ready": (None, "raw_beef"),
            "rack_ready": (None, "raw_beef"),
            "raw_ready": ("raw_beef", None),
            "grill_approach_ready": ("raw_beef", None),
            "beef_carry_ready": ("raw_beef", None),
            "beef_pick_ready": (None, None),
            "wash_ready": (None, None),
            "wash_required_ready": (None, "raw_beef"),
            "sink_drop_ready": ("dirty_plate", "raw_beef"),
            "dirty_plate_carry_ready": ("dirty_plate", "raw_beef"),
            "dirty_return_ready": (None, None),
            "second_dish_ready": (None, "raw_beef"),
            "plate_exhausted_ready": (None, "raw_beef"),
            "fire_recovery_ready": ("clean_plate", "burnt_beef"),
            "fire_plate_drop_ready": ("clean_plate", "burnt_beef"),
            "fire_plate_parked_ready": (None, "burnt_beef"),
            "fire_extinguisher_pick_ready": (None, "burnt_beef"),
            "fire_extinguisher_carry_ready": (
                "extinguisher",
                "burnt_beef",
            ),
            "fire_suppress_ready": ("extinguisher", "burnt_beef"),
        }
        for stage, (held, grill_food) in expected.items():
            with self.subTest(stage=stage):
                state = mdp.get_start_state(1, stage=stage)
                mdp.validate_state(state)
                self.assertEqual(state.players[0].held_object, held)
                self.assertEqual(state.grill.food, grill_food)
                self.assertEqual(
                    state.delivered_orders,
                    (
                        state.total_plates
                        if stage
                        in {
                            "wash_required_ready",
                            "sink_drop_ready",
                            "dirty_plate_carry_ready",
                            "plate_exhausted_ready",
                        }
                        else int(stage == "second_dish_ready")
                    ),
                )

        parked = mdp.get_start_state(
            1, stage="fire_plate_parked_ready"
        )
        self.assertEqual(parked.players[0].position, (4, 1))
        self.assertEqual(
            parked.counter_objects,
            {(4, 0): "plate_bun_lettuce"},
        )

        wash_ready = mdp.get_start_state(1, stage="wash_ready")
        self.assertTrue(wash_ready.sink.has_dirty_plate)
        self.assertEqual(wash_ready.clean_plates, 3)
        wash_required = mdp.get_start_state(
            1, stage="wash_required_ready"
        )
        self.assertTrue(wash_required.sink.has_dirty_plate)
        self.assertEqual(wash_required.clean_plates, 0)
        self.assertEqual(
            wash_required.dirty_plates_at_return,
            wash_required.total_plates - 1,
        )
        sink_drop = mdp.get_start_state(1, stage="sink_drop_ready")
        self.assertEqual(
            sink_drop.players[0].held_object,
            "dirty_plate",
        )
        self.assertEqual(sink_drop.clean_plates, 0)
        dirty_carry = mdp.get_start_state(
            1, stage="dirty_plate_carry_ready"
        )
        self.assertEqual(
            dirty_carry.players[0].held_object,
            "dirty_plate",
        )
        self.assertNotEqual(
            sink_drop.players[0].position,
            dirty_carry.players[0].position,
        )
        dirty_return_ready = mdp.get_start_state(
            1, stage="dirty_return_ready"
        )
        self.assertEqual(dirty_return_ready.dirty_plates_at_return, 1)
        second_dish_ready = mdp.get_start_state(
            1, stage="second_dish_ready"
        )
        self.assertEqual(second_dish_ready.delivered_orders, 1)
        self.assertEqual(second_dish_ready.dirty_plates_at_return, 1)
        self.assertEqual(second_dish_ready.grill.food, "raw_beef")
        plate_exhausted = mdp.get_start_state(
            1, stage="plate_exhausted_ready"
        )
        self.assertEqual(plate_exhausted.clean_plates, 0)
        self.assertEqual(
            plate_exhausted.dirty_plates_at_return,
            plate_exhausted.total_plates,
        )
        self.assertEqual(
            plate_exhausted.delivered_orders,
            plate_exhausted.total_plates,
        )
        self.assertEqual(plate_exhausted.grill.food, "raw_beef")

        beef_pick_ready = mdp.get_start_state(
            1, stage="beef_pick_ready"
        )
        beef_carry_ready = mdp.get_start_state(
            1, stage="beef_carry_ready"
        )
        grill_approach_ready = mdp.get_start_state(
            1, stage="grill_approach_ready"
        )
        self.assertEqual(
            grill_approach_ready.players[0].position, (4, 1)
        )
        self.assertEqual(
            beef_carry_ready.players[0].position, (3, 1)
        )
        self.assertEqual(
            beef_carry_ready.players[0].held_object, "raw_beef"
        )
        self.assertEqual(beef_pick_ready.players[0].position, (3, 1))
        picked = mdp.get_state_transition(
            beef_pick_ready, [Action.PICK_DROP]
        )
        self.assertEqual(
            picked.state.players[0].held_object, "raw_beef"
        )
        self.assertIn(
            "dispenser_pickup",
            {event["type"] for event in picked.info["events"]},
        )

        with self.assertRaisesRegex(ValueError, "one active agent"):
            mdp.get_start_state(2, stage="serve_ready")
        with self.assertRaisesRegex(ValueError, "Unknown curriculum"):
            mdp.get_start_state(1, stage="invented")

    def test_rejects_unknown_or_wrong_sized_joint_actions(self):
        mdp = BurgerGridworld()
        state = mdp.get_standard_start_state(2)
        with self.assertRaises(ValueError):
            mdp.get_state_transition(state, [Action.STAY])
        with self.assertRaises(ValueError):
            mdp.get_state_transition(state, [Action.STAY, "teleport"])

    def test_adjacent_interact_does_not_require_facing(self):
        mdp = BurgerGridworld(layout(("B  ", "   "), ((1, 0),)))
        state = mdp.get_standard_start_state(1)
        state.players[0].orientation = Direction.SOUTH
        picked = mdp.get_state_transition(state, [Action.PICK_DROP]).state
        self.assertEqual(picked.players[0].held_object, "bun")
        self.assertEqual(picked.players[0].orientation, Direction.SOUTH)

    def test_extinguisher_can_be_returned_to_its_station(self):
        mdp = BurgerGridworld(
            layout((" E ", "   "), ((1, 1),))
        )
        state = mdp.get_standard_start_state(1)
        state.players[0].orientation = Direction.NORTH

        self.assertTrue(
            mdp.context_action_available(state, 0, Action.PICK_DROP)
        )
        holding = mdp.get_state_transition(
            state, [Action.PICK_DROP]
        ).state
        self.assertEqual(
            holding.players[0].held_object, "extinguisher"
        )
        self.assertFalse(holding.extinguisher_available)
        self.assertTrue(
            mdp.context_action_available(
                holding, 0, Action.PICK_DROP
            )
        )
        returned = mdp.get_state_transition(
            holding, [Action.PICK_DROP]
        )
        self.assertIsNone(returned.state.players[0].held_object)
        self.assertTrue(returned.state.extinguisher_available)
        self.assertIn(
            "extinguisher_return",
            {event["type"] for event in returned.info["events"]},
        )

    def test_extinguisher_is_one_conserved_physical_object(self):
        mdp = BurgerGridworld(
            layout(("   ", " E ", "   "), ((0, 1), (2, 1)))
        )
        state = mdp.get_standard_start_state(2)

        picked = mdp.get_state_transition(
            state, [Action.PICK_DROP, Action.PICK_DROP]
        )

        self.assertEqual(
            [
                player.held_object
                for player in picked.state.players
            ].count("extinguisher"),
            1,
        )
        self.assertFalse(picked.state.extinguisher_available)
        self.assertEqual(
            [
                event["type"]
                for event in picked.info["events"]
            ].count("extinguisher_pickup"),
            1,
        )
        mdp.validate_state(picked.state)

        forged = copy.deepcopy(picked.state)
        second_player = next(
            player
            for player in forged.players
            if player.held_object != "extinguisher"
        )
        second_player.held_object = "extinguisher"
        with self.assertRaisesRegex(
            ValueError, "Extinguisher conservation"
        ):
            mdp.validate_state(forged)

    def test_clean_plates_stack_only_at_plate_rack(self):
        mdp = BurgerGridworld(
            layout(("   ", " D ", "   "), ((0, 1), (2, 1)))
        )
        state = BurgerState(
            players=[
                BurgerPlayerState(
                    (0, 1), Direction.NORTH, "clean_plate"
                ),
                BurgerPlayerState(
                    (2, 1), Direction.NORTH, "clean_plate"
                ),
            ],
            clean_plates=2,
            total_plates=4,
        )

        stacked = mdp.get_state_transition(
            state, [Action.PICK_DROP, Action.PICK_DROP]
        ).state
        self.assertEqual(stacked.clean_plates, 4)
        self.assertTrue(
            all(
                player.held_object is None
                for player in stacked.players
            )
        )

        taken = mdp.get_state_transition(
            stacked, [Action.PICK_DROP, Action.PICK_DROP]
        ).state
        self.assertEqual(taken.clean_plates, 2)
        self.assertTrue(
            all(
                player.held_object == "clean_plate"
                for player in taken.players
            )
        )
        mdp.validate_state(taken)

    def test_objects_cannot_be_dropped_on_floor_or_disappear(self):
        mdp = BurgerGridworld(
            layout(("   ", "   "), ((0, 0),)),
            BurgerConfig(total_plates=1),
        )
        state = BurgerState(
            players=[
                BurgerPlayerState(
                    (0, 0), Direction.EAST, "plate_bun"
                )
            ],
            clean_plates=0,
            total_plates=1,
        )
        unchanged = mdp.get_state_transition(
            state, [Action.PICK_DROP]
        ).state
        self.assertEqual(
            unchanged.players[0].held_object, "plate_bun"
        )
        self.assertEqual(unchanged.counter_objects, {})
        mdp.validate_state(unchanged)

    def test_trash_discards_loose_food_without_positive_event_reward(self):
        mdp = BurgerGridworld(layout((" T ", "   "), ((0, 0),)))
        state = mdp.get_standard_start_state(1)
        state.players[0].orientation = Direction.EAST
        state.players[0].held_object = "raw_beef"

        transition = mdp.get_state_transition(state, [Action.PICK_DROP])

        self.assertIsNone(transition.state.players[0].held_object)
        self.assertIn(
            "food_discarded",
            {event["type"] for event in transition.info["events"]},
        )
        self.assertLess(transition.reward, 0.0)
        self.assertNotIn(
            "food_discarded", transition.info["reward_breakdown"]
        )

    def test_dispenser_to_trash_cycle_cannot_farm_shaping_reward(self):
        mdp = BurgerGridworld(layout(("B T", "   "), ((1, 0),)))
        state = mdp.get_standard_start_state(1)
        state.players[0].orientation = Direction.WEST
        cycle_reward = 0.0

        for _ in range(20):
            for action in (
                Action.PICK_DROP,
                Direction.EAST,
                Action.PICK_DROP,
                Direction.WEST,
            ):
                transition = mdp.get_state_transition(state, [action])
                state = transition.state
                cycle_reward += transition.reward

        self.assertLess(cycle_reward, 0.0)
        self.assertIsNone(state.players[0].held_object)

    def test_counter_drop_pick_cycle_cannot_farm_shaping_reward(self):
        mdp = BurgerGridworld()
        state = mdp.get_standard_start_state(1)
        state.players[0].position = (4, 6)
        state.players[0].held_object = "bun"
        mdp.validate_state(state)
        cycle_reward = 0.0

        for _ in range(20):
            transition = mdp.get_state_transition(
                state, [Action.PICK_DROP]
            )
            state = transition.state
            cycle_reward += transition.reward

        self.assertLess(cycle_reward, 0.0)
        self.assertEqual(state.players[0].held_object, "bun")
        self.assertNotIn((4, 7), state.counter_objects)

    def test_wash_rewards_require_one_conserved_dirty_plate(self):
        mdp = BurgerGridworld(
            config=BurgerConfig(total_plates=1, wash_steps=2)
        )
        state = mdp.get_start_state(1, stage="standard")
        state.clean_plates = 0
        state.sink = SinkState(has_dirty_plate=True, wash_progress=0)
        state.players[0].position = (0, 6)
        mdp.validate_state(state)

        first = mdp.get_state_transition(state, [Action.PROCESS])
        self.assertEqual(
            first.info["reward_breakdown"]["wash_progress"],
            mdp.config.reward.wash_progress,
        )
        completed = mdp.get_state_transition(
            first.state, [Action.PROCESS]
        )
        self.assertEqual(
            completed.info["reward_breakdown"]["plate_washed"],
            mdp.config.reward.plate_washed,
        )
        self.assertEqual(
            completed.state.players[0].held_object,
            "clean_plate",
        )

        repeated = mdp.get_state_transition(
            completed.state, [Action.PROCESS]
        )
        self.assertEqual(
            repeated.info["reward_breakdown"]["wash_progress"],
            0.0,
        )
        self.assertEqual(
            repeated.info["reward_breakdown"]["plate_washed"],
            0.0,
        )

    def test_trash_empties_loaded_plate_but_conserves_the_plate(self):
        mdp = BurgerGridworld(
            layout((" T ", "   "), ((0, 0),)),
            BurgerConfig(total_plates=1),
        )
        for loaded_plate in sorted(
            item
            for item, contents in PLATE_CONTENTS.items()
            if contents
        ):
            with self.subTest(loaded_plate=loaded_plate):
                state = BurgerState(
                    players=[
                        BurgerPlayerState(
                            (0, 0), Direction.EAST, loaded_plate
                        )
                    ],
                    clean_plates=0,
                    total_plates=1,
                )

                transition = mdp.get_state_transition(
                    state, [Action.PICK_DROP]
                )

                self.assertEqual(
                    transition.state.players[0].held_object,
                    "clean_plate",
                )
                self.assertIn(
                    "plate_contents_discarded",
                    {
                        event["type"]
                        for event in transition.info["events"]
                    },
                )
                mdp.validate_state(transition.state)

    def test_trash_cannot_delete_clean_dirty_plates_or_extinguisher(self):
        mdp = BurgerGridworld(layout((" T ", "   "), ((0, 0),)))
        for item in ("clean_plate", "dirty_plate", "extinguisher"):
            with self.subTest(item=item):
                state = mdp.get_standard_start_state(1)
                state.players[0].orientation = Direction.EAST
                state.players[0].held_object = item
                if item in {"clean_plate", "dirty_plate"}:
                    state.clean_plates = 3
                if item == "extinguisher":
                    state.extinguisher_available = False
                transition = mdp.get_state_transition(
                    state, [Action.PICK_DROP]
                )
                self.assertEqual(
                    transition.state.players[0].held_object, item
                )
                self.assertFalse(
                    any(
                        event["type"]
                        in {"food_discarded", "plate_contents_discarded"}
                        for event in transition.info["events"]
                    )
                )

    def test_plate_can_be_placed_on_and_retrieved_from_counter(self):
        mdp = BurgerGridworld(
            layout((" X ", "   "), ((0, 0),)),
            BurgerConfig(total_plates=1),
        )
        state = BurgerState(
            players=[
                BurgerPlayerState(
                    (0, 0), Direction.EAST, "plate_bun"
                )
            ],
            clean_plates=0,
            total_plates=1,
        )
        placed = mdp.get_state_transition(
            state, [Action.PICK_DROP]
        ).state
        self.assertIsNone(placed.players[0].held_object)
        self.assertEqual(
            placed.counter_objects[(1, 0)], "plate_bun"
        )
        retrieved = mdp.get_state_transition(
            placed, [Action.PICK_DROP]
        ).state
        self.assertEqual(
            retrieved.players[0].held_object, "plate_bun"
        )
        self.assertNotIn((1, 0), retrieved.counter_objects)
        mdp.validate_state(retrieved)

    def test_occupied_counter_rejects_second_item_without_overwrite(self):
        mdp = BurgerGridworld(
            layout((" X", "  "), ((0, 0),)),
            BurgerConfig(total_plates=1),
        )
        state = BurgerState(
            players=[
                BurgerPlayerState(
                    (0, 0), Direction.EAST, "lettuce"
                )
            ],
            counter_objects={(1, 0): "bun"},
            clean_plates=1,
            total_plates=1,
        )

        blocked = mdp.get_state_transition(
            state, [Action.PICK_DROP]
        )
        self.assertEqual(
            blocked.state.counter_objects, {(1, 0): "bun"}
        )
        self.assertEqual(
            blocked.state.players[0].held_object, "lettuce"
        )
        self.assertFalse(
            any(
                event["type"] in {"counter_drop", "counter_pickup"}
                for event in blocked.info["events"]
            )
        )
        mdp.validate_state(blocked.state)

    def test_counter_pickup_is_atomic_under_simultaneous_interact(self):
        mdp = BurgerGridworld(layout((" X ", "   "), ((0, 0), (2, 0))))
        state = mdp.get_standard_start_state(2)
        state.players[0].orientation = Direction.EAST
        state.players[1].orientation = Direction.WEST
        state.counter_objects[(1, 0)] = "bun"

        next_state = mdp.get_state_transition(
            state, [Action.PICK_DROP, Action.PICK_DROP]
        ).state
        self.assertEqual(next_state.players[0].held_object, "bun")
        self.assertIsNone(next_state.players[1].held_object)
        self.assertNotIn((1, 0), next_state.counter_objects)

    def test_held_plate_collects_counter_items_without_losing_plate(self):
        mdp = BurgerGridworld(
            layout((" X ", "   "), ((0, 0),)),
            BurgerConfig(total_plates=1),
        )
        state = BurgerState(
            players=[
                BurgerPlayerState(
                    (0, 0), Direction.EAST, "clean_plate"
                )
            ],
            counter_objects={(1, 0): "bun"},
            clean_plates=0,
            total_plates=1,
        )
        with_bun = mdp.get_state_transition(
            state, [Action.PICK_DROP]
        ).state
        self.assertEqual(with_bun.players[0].held_object, "plate_bun")
        self.assertNotIn((1, 0), with_bun.counter_objects)

        with_bun.counter_objects[(1, 0)] = "lettuce"
        with_lettuce = mdp.get_state_transition(
            with_bun, [Action.PICK_DROP]
        ).state
        self.assertEqual(
            with_lettuce.players[0].held_object,
            "plate_bun_lettuce",
        )
        self.assertNotIn((1, 0), with_lettuce.counter_objects)
        mdp.validate_state(with_lettuce)

    def test_held_plate_collects_cooked_beef_from_grill(self):
        mdp = BurgerGridworld(
            layout((" G ", "   "), ((0, 0),)),
            BurgerConfig(total_plates=1),
        )
        state = BurgerState(
            players=[
                BurgerPlayerState(
                    (0, 0),
                    Direction.EAST,
                    "plate_bun_lettuce",
                )
            ],
            clean_plates=0,
            total_plates=1,
            grill=GrillState(food="cooked_beef", cook_ticks=12),
        )
        plated = mdp.get_state_transition(
            state, [Action.PICK_DROP]
        ).state
        self.assertEqual(
            plated.players[0].held_object, "plated_burger"
        )
        self.assertIsNone(plated.grill.food)
        mdp.validate_state(plated)

    def test_cooked_beef_cannot_be_picked_up_without_a_plate(self):
        mdp = BurgerGridworld(
            layout((" G ", "   "), ((0, 0),)),
            BurgerConfig(total_plates=1),
        )
        state = BurgerState(
            players=[BurgerPlayerState((0, 0), Direction.SOUTH)],
            clean_plates=1,
            total_plates=1,
            grill=GrillState(food="cooked_beef", cook_ticks=12),
        )

        self.assertFalse(
            mdp.context_action_available(state, 0, Action.PICK_DROP)
        )
        untouched = mdp.get_state_transition(
            state, [Action.PICK_DROP]
        ).state
        self.assertIsNone(untouched.players[0].held_object)
        self.assertEqual(untouched.grill.food, "cooked_beef")

        untouched.players[0].held_object = "clean_plate"
        untouched.clean_plates = 0
        plated = mdp.get_state_transition(
            untouched, [Action.PICK_DROP]
        ).state
        self.assertEqual(
            plated.players[0].held_object,
            "plate_cooked_beef",
        )
        self.assertIsNone(plated.grill.food)

    def test_cooked_beef_on_counter_cannot_be_picked_up_barehanded(self):
        mdp = BurgerGridworld(
            layout((" X ", "   "), ((0, 0),)),
            BurgerConfig(total_plates=1),
        )
        state = BurgerState(
            players=[BurgerPlayerState((0, 0), Direction.SOUTH)],
            counter_objects={(1, 0): "cooked_beef"},
            clean_plates=1,
            total_plates=1,
        )

        untouched = mdp.get_state_transition(
            state, [Action.PICK_DROP]
        ).state
        self.assertIsNone(untouched.players[0].held_object)
        self.assertEqual(
            untouched.counter_objects[(1, 0)],
            "cooked_beef",
        )

        untouched.players[0].held_object = "clean_plate"
        untouched.clean_plates = 0
        plated = mdp.get_state_transition(
            untouched, [Action.PICK_DROP]
        ).state
        self.assertEqual(
            plated.players[0].held_object,
            "plate_cooked_beef",
        )
        self.assertNotIn((1, 0), plated.counter_objects)

    def test_held_partial_plate_collects_ingredient_from_dispenser(self):
        mdp = BurgerGridworld(
            layout(("B  ", "   "), ((0, 1),)),
            BurgerConfig(total_plates=1),
        )
        state = BurgerState(
            players=[
                BurgerPlayerState(
                    (0, 1),
                    Direction.NORTH,
                    "plate_lettuce",
                )
            ],
            clean_plates=0,
            total_plates=1,
        )

        result = mdp.get_state_transition(
            state, [Action.PICK_DROP]
        )

        self.assertEqual(
            result.state.players[0].held_object,
            "plate_bun_lettuce",
        )
        self.assertIn(
            "ingredient_added_from_dispenser",
            [event["type"] for event in result.info["events"]],
        )
        self.assertTrue(
            mdp.context_action_available(
                state, 0, Action.PICK_DROP
            )
        )
        mdp.validate_state(result.state)

    def test_food_cannot_assemble_without_a_physical_plate(self):
        mdp = BurgerGridworld(
            layout((" L ", "   "), ((1, 1),)),
            BurgerConfig(total_plates=1),
        )
        state = BurgerState(
            players=[
                BurgerPlayerState(
                    (1, 1), Direction.NORTH, "bun"
                )
            ],
            clean_plates=0,
            dirty_plates_at_return=1,
            total_plates=1,
        )

        unchanged = mdp.get_state_transition(
            state, [Action.PICK_DROP]
        ).state

        self.assertEqual(unchanged.players[0].held_object, "bun")
        self.assertFalse(
            mdp.context_action_available(
                state, 0, Action.PICK_DROP
            )
        )
        mdp.validate_state(unchanged)

    def test_partial_burger_is_assembled_on_one_visible_counter(self):
        mdp = BurgerGridworld(
            layout(
                ("   ", " X ", "   "),
                ((1, 0),),
            )
        )
        state = BurgerState(
            players=[
                BurgerPlayerState(
                    (1, 0), Direction.SOUTH, "clean_plate"
                ),
            ],
            clean_plates=0,
            total_plates=1,
        )
        counter = (1, 1)
        expected = (
            ("clean_plate", "clean_plate"),
            ("bun", "plate_bun"),
            ("lettuce", "plate_bun_lettuce"),
        )
        for held, counter_item in expected:
            state.players[0].held_object = held
            state = mdp.get_state_transition(
                state, [Action.PICK_DROP]
            ).state
            self.assertIsNone(state.players[0].held_object)
            self.assertEqual(
                state.counter_objects[counter], counter_item
            )
            self.assertEqual(len(state.counter_objects), 1)

        assembled = mdp.get_state_transition(
            state, [Action.PICK_DROP]
        ).state
        self.assertEqual(
            assembled.players[0].held_object, "plate_bun_lettuce"
        )
        self.assertNotIn(counter, assembled.counter_objects)

    def test_duplicate_counter_component_cannot_be_consumed_twice(self):
        mdp = BurgerGridworld(
            layout(("   ", " X ", "   "), ((1, 0),))
        )
        state = BurgerState(
            players=[
                BurgerPlayerState((1, 0), Direction.SOUTH, "bun"),
            ],
            counter_objects={(1, 1): "plate_bun"},
            clean_plates=0,
            total_plates=1,
        )
        next_state = mdp.get_state_transition(
            state, [Action.PICK_DROP]
        ).state
        self.assertEqual(
            next_state.counter_objects[(1, 1)], "plate_bun"
        )
        self.assertEqual(next_state.players[0].held_object, "bun")

    def test_delivery_reward_is_event_sourced_and_single_use(self):
        mdp = BurgerGridworld(
            layout((" S", "  "), ((0, 0),)),
            BurgerConfig(total_plates=1, plate_return_steps=2),
        )
        state = BurgerState(
            players=[
                BurgerPlayerState((0, 0), Direction.EAST, "plated_burger")
            ],
            clean_plates=0,
            total_plates=1,
        )
        delivered = mdp.get_state_transition(state, [Action.PICK_DROP])
        self.assertEqual(delivered.state.delivered_orders, 1)
        self.assertEqual(
            delivered.info["reward_breakdown"]["correct_delivery"], 20.0
        )
        self.assertEqual(len(delivered.state.pending_plate_returns), 1)

        repeated = mdp.get_state_transition(
            delivered.state, [Action.PICK_DROP]
        )
        self.assertEqual(repeated.state.delivered_orders, 1)
        self.assertEqual(
            repeated.info["reward_breakdown"]["correct_delivery"], 0.0
        )
        self.assertEqual(repeated.state.dirty_plates_at_return, 1)

    def test_delivery_and_delayed_dish_return_use_separate_hatches(self):
        mdp = BurgerGridworld(
            BurgerLayout(
                rows=(
                    "R SXXXXXXXX",
                    "           ",
                    "GGGGGGGGGGG",
                    "BLMDWETXXXX",
                ),
                player_starts=((0, 1),),
                name="separate_hatches",
            ),
            BurgerConfig(total_plates=1, plate_return_steps=3),
        )
        state = BurgerState(
            players=[
                BurgerPlayerState(
                    (0, 1), Direction.WEST, "plated_burger"
                )
            ],
            clean_plates=0,
            total_plates=1,
        )

        wrong_hatch = mdp.get_state_transition(
            state, [Action.PICK_DROP]
        )
        self.assertEqual(wrong_hatch.state.delivered_orders, 0)
        self.assertEqual(
            wrong_hatch.state.players[0].held_object,
            "plated_burger",
        )

        toward_serve = mdp.get_state_transition(
            wrong_hatch.state, [Direction.EAST]
        )
        beside_serve = mdp.get_state_transition(
            toward_serve.state, [Direction.EAST]
        )
        delivered = mdp.get_state_transition(
            beside_serve.state, [Action.PICK_DROP]
        )
        self.assertEqual(delivered.state.delivered_orders, 1)
        self.assertEqual(delivered.state.dirty_plates_at_return, 0)
        self.assertEqual(len(delivered.state.pending_plate_returns), 1)

        before_due = mdp.get_state_transition(
            delivered.state, [Action.STAY]
        )
        self.assertEqual(before_due.state.dirty_plates_at_return, 0)
        returned = mdp.get_state_transition(
            before_due.state, [Action.STAY]
        )
        self.assertEqual(returned.state.dirty_plates_at_return, 1)
        self.assertEqual(returned.state.pending_plate_returns, [])

        wrong_pickup = mdp.get_state_transition(
            returned.state, [Action.PICK_DROP]
        )
        self.assertIsNone(wrong_pickup.state.players[0].held_object)
        self.assertEqual(wrong_pickup.state.dirty_plates_at_return, 1)

        toward_return = mdp.get_state_transition(
            wrong_pickup.state, [Direction.WEST]
        )
        beside_return = mdp.get_state_transition(
            toward_return.state, [Direction.WEST]
        )
        picked = mdp.get_state_transition(
            beside_return.state, [Action.PICK_DROP]
        )
        self.assertEqual(
            picked.state.players[0].held_object, "dirty_plate"
        )
        self.assertEqual(picked.state.dirty_plates_at_return, 0)

    def test_potential_shaping_telescopes_without_cycle_bonus(self):
        mdp = BurgerGridworld()
        state = mdp.get_standard_start_state(1)
        initial_potential = mdp.potential(state)
        actions = [Action.PICK_DROP] + [Action.STAY] * 12
        shaping_terms = []
        for action in actions:
            transition = mdp.get_state_transition(state, [action])
            shaping_terms.append(
                transition.info["reward_breakdown"]["potential"]
            )
            state = transition.state

        gamma = mdp.config.reward.gamma
        discounted_shaping = sum(
            gamma**step * value
            for step, value in enumerate(shaping_terms)
        )
        expected = (
            -initial_potential
            + gamma ** len(actions) * mdp.potential(state)
        )
        self.assertAlmostEqual(
            discounted_shaping, expected, places=10
        )

    def test_fire_recovery_potential_rewards_every_required_phase(self):
        mdp = BurgerGridworld()
        holding_plate = mdp.get_start_state(1)
        holding_plate.players[0].position = (6, 1)
        holding_plate.players[0].held_object = "clean_plate"
        holding_plate.clean_plates -= 1
        holding_plate.grill = GrillState(food="burnt_beef")
        mdp.validate_state(holding_plate)

        plate_dropped = copy.deepcopy(holding_plate)
        plate_dropped.players[0].held_object = None
        plate_dropped.counter_objects[(2, 0)] = "clean_plate"
        mdp.validate_state(plate_dropped)

        extinguisher_held = copy.deepcopy(plate_dropped)
        extinguisher_held.players[0].held_object = "extinguisher"
        extinguisher_held.extinguisher_available = False
        mdp.validate_state(extinguisher_held)

        fire_cleared = copy.deepcopy(extinguisher_held)
        fire_cleared.grill = GrillState()
        mdp.validate_state(fire_cleared)

        extinguisher_returned = copy.deepcopy(fire_cleared)
        extinguisher_returned.players[0].held_object = None
        extinguisher_returned.extinguisher_available = True
        mdp.validate_state(extinguisher_returned)

        potentials = [
            mdp.potential(state)
            for state in (
                holding_plate,
                plate_dropped,
                extinguisher_held,
                fire_cleared,
                extinguisher_returned,
            )
        ]
        self.assertEqual(potentials, sorted(potentials))
        self.assertEqual(len(set(potentials)), len(potentials))

    def test_fire_plate_drop_pickup_cycle_cannot_earn_shaping_reward(self):
        mdp = BurgerGridworld()
        holding = mdp.get_start_state(
            1, stage="fire_plate_drop_ready"
        )
        dropped = mdp.get_state_transition(
            holding, [Action.PICK_DROP]
        )
        picked_back = mdp.get_state_transition(
            dropped.state, [Action.PICK_DROP]
        )

        self.assertEqual(
            dropped.state.players[0].held_object, None
        )
        self.assertEqual(
            picked_back.state.players[0].held_object, "clean_plate"
        )
        discounted_cycle_reward = (
            dropped.info["reward_breakdown"]["potential"]
            + mdp.config.reward.gamma
            * picked_back.info["reward_breakdown"]["potential"]
        )
        self.assertLess(discounted_cycle_reward, 0.0)

    def test_dirty_plate_on_counter_is_not_washed_or_recipe_progress(self):
        mdp = BurgerGridworld(
            config=BurgerConfig(
                total_plates=2,
                reward=BurgerRewardConfig(
                    navigation_potential_scale=0.0
                ),
            )
        )
        held_dirty = BurgerState(
            players=[
                BurgerPlayerState((2, 1), Direction.NORTH, "dirty_plate")
            ],
            clean_plates=0,
            total_plates=2,
        )
        counter_dirty = copy.deepcopy(held_dirty)
        counter_dirty.players[0].held_object = None
        counter_dirty.counter_objects[(2, 0)] = "dirty_plate"

        self.assertGreater(
            mdp.potential(held_dirty),
            mdp.potential(counter_dirty),
        )

        no_plate = BurgerState(
            players=[BurgerPlayerState((2, 1), Direction.NORTH)],
            clean_plates=1,
            total_plates=2,
            grill=GrillState(food="raw_beef"),
        )
        dirty_during_cooking = copy.deepcopy(no_plate)
        dirty_during_cooking.players[0].held_object = "dirty_plate"
        self.assertEqual(
            mdp.potential(no_plate),
            mdp.potential(dirty_during_cooking),
        )

    def test_dirty_plate_counter_cycle_receives_handling_cost(self):
        mdp = BurgerGridworld()
        state = mdp.get_start_state(
            1, stage="dirty_plate_carry_ready"
        )
        state.players[0].position = (2, 1)

        dropped = mdp.get_state_transition(state, [Action.PICK_DROP])
        picked_up = mdp.get_state_transition(
            dropped.state, [Action.PICK_DROP]
        )

        self.assertEqual(
            dropped.info["reward_breakdown"][
                "dirty_plate_counter_handling"
            ],
            -0.25,
        )
        self.assertEqual(
            picked_up.info["reward_breakdown"][
                "dirty_plate_counter_handling"
            ],
            -0.25,
        )

    def test_partial_plate_with_empty_grill_navigates_to_empty_counter(self):
        mdp = BurgerGridworld()
        state = mdp.get_standard_start_state(1)
        state.players[0].held_object = "plate_bun_lettuce"
        state.clean_plates -= 1
        for counter in mdp._counter_distances:
            if counter != (2, 0):
                state.counter_objects[counter] = "bun"

        state.players[0].position = (7, 6)
        far_from_counter = mdp._single_agent_navigation_potential(state)
        state.players[0].position = (2, 1)
        beside_counter = mdp._single_agent_navigation_potential(state)
        self.assertGreater(beside_counter, far_from_counter)

    def test_parking_partial_plate_advances_workflow_without_cycle_bonus(self):
        mdp = BurgerGridworld()
        state = mdp.get_standard_start_state(1)
        state.players[0].position = (2, 1)
        state.players[0].orientation = Direction.NORTH
        state.players[0].held_object = "plate_bun_lettuce"
        state.clean_plates -= 1
        mdp.validate_state(state)

        dropped = mdp.get_state_transition(state, [Action.PICK_DROP])
        picked_back = mdp.get_state_transition(
            dropped.state, [Action.PICK_DROP]
        )

        self.assertEqual(
            dropped.state.counter_objects[(2, 0)],
            "plate_bun_lettuce",
        )
        self.assertGreater(
            dropped.info["reward_breakdown"]["potential"], 0.0
        )
        gamma = mdp.config.reward.gamma
        discounted_cycle = (
            dropped.info["reward_breakdown"]["potential"]
            + gamma
            * picked_back.info["reward_breakdown"]["potential"]
        )
        self.assertAlmostEqual(
            discounted_cycle,
            gamma**2 * mdp.potential(state) - mdp.potential(state),
        )
        self.assertLess(discounted_cycle, 0.0)

    def test_fire_navigation_frees_hand_and_returns_extinguisher(self):
        mdp = BurgerGridworld()
        holding_plate = mdp.get_start_state(
            1, stage="fire_recovery_ready"
        )
        for counter in mdp._counter_distances:
            if counter != (2, 0):
                holding_plate.counter_objects[counter] = "bun"

        holding_plate.players[0].position = (7, 6)
        far_from_empty_counter = mdp._single_agent_navigation_potential(
            holding_plate
        )
        holding_plate.players[0].position = (2, 1)
        beside_empty_counter = mdp._single_agent_navigation_potential(
            holding_plate
        )
        self.assertGreater(
            beside_empty_counter, far_from_empty_counter
        )

        fire_cleared = mdp.get_start_state(
            1, stage="fire_suppress_ready"
        )
        fire_cleared.grill = GrillState()
        fire_cleared.players[0].position = (0, 6)
        far_from_extinguisher = (
            mdp._single_agent_navigation_potential(fire_cleared)
        )
        fire_cleared.players[0].position = (6, 1)
        beside_extinguisher = (
            mdp._single_agent_navigation_potential(fire_cleared)
        )
        self.assertGreater(
            beside_extinguisher, far_from_extinguisher
        )

    def test_navigation_retrieves_parked_plate_when_rack_is_empty(self):
        mdp = BurgerGridworld(config=BurgerConfig(total_plates=1))
        state = mdp.get_standard_start_state(1)
        state.clean_plates = 0
        state.counter_objects[(2, 0)] = "clean_plate"
        state.grill = GrillState(food="raw_beef")
        mdp.validate_state(state)

        state.players[0].position = (7, 6)
        far_from_plate = mdp._single_agent_navigation_potential(state)
        state.players[0].position = (2, 1)
        beside_plate = mdp._single_agent_navigation_potential(state)
        self.assertGreater(beside_plate, far_from_plate)

    def test_every_step_reward_is_auditable_and_pickup_has_no_event_bonus(self):
        mdp = BurgerGridworld(layout((" M", "  "), ((0, 0),)))
        state = mdp.get_standard_start_state(1)
        picked = mdp.get_state_transition(state, [Action.PICK_DROP])
        breakdown = picked.info["reward_breakdown"]

        self.assertEqual(
            set(breakdown),
            {
                "correct_delivery",
                "raw_beef_placed",
                "dirty_plate_pickup",
                "wash_started",
                "wash_progress",
                "plate_washed",
                "fire_extinguished",
                "fire_started",
                "fire_active",
                "fire_food_handling",
                "dirty_plate_counter_handling",
                "collision",
                "time_step",
                "potential",
            },
        )
        self.assertAlmostEqual(picked.reward, sum(breakdown.values()))
        self.assertEqual(breakdown["correct_delivery"], 0.0)
        self.assertEqual(breakdown["fire_started"], 0.0)
        self.assertEqual(breakdown["fire_active"], 0.0)
        self.assertEqual(breakdown["collision"], 0.0)
        self.assertEqual(breakdown["time_step"], 0.0)
        self.assertAlmostEqual(
            breakdown["potential"],
            mdp.config.reward.gamma * mdp.potential(picked.state)
            - mdp.potential(state),
        )
        self.assertIn(
            "dispenser_pickup",
            {event["type"] for event in picked.info["events"]},
        )
        idle = mdp.get_state_transition(picked.state, [Action.STAY])
        self.assertLess(idle.reward, 0.0)
        self.assertAlmostEqual(
            idle.reward,
            sum(idle.info["reward_breakdown"].values()),
        )

    def test_navigation_shaping_rewards_progress_without_cycle_bonus(self):
        mdp = BurgerGridworld()
        state = mdp.get_standard_start_state(1)
        closer = mdp.get_state_transition(state, [Direction.EAST])
        farther = mdp.get_state_transition(
            closer.state, [Direction.WEST]
        )

        self.assertGreater(
            closer.info["reward_breakdown"]["potential"], 0.0
        )
        gamma = mdp.config.reward.gamma
        discounted = (
            closer.info["reward_breakdown"]["potential"]
            + gamma * farther.info["reward_breakdown"]["potential"]
        )
        expected = (
            gamma**2 * mdp.potential(farther.state)
            - mdp.potential(state)
        )
        self.assertAlmostEqual(discounted, expected)

    def test_navigation_prioritizes_next_order_until_plates_are_exhausted(self):
        mdp = BurgerGridworld()
        reusable_plates = mdp.get_standard_start_state(1)
        reusable_plates.clean_plates = 3
        reusable_plates.pending_plate_returns = [12]
        reusable_plates.delivered_orders = 1
        reusable_plates.players[0].position = (3, 1)
        at_beef = mdp._single_agent_navigation_potential(reusable_plates)
        reusable_plates.players[0].position = (6, 6)
        at_return = mdp._single_agent_navigation_potential(reusable_plates)
        self.assertGreater(at_beef, at_return)

        exhausted = mdp.get_start_state(
            1, stage="plate_exhausted_ready"
        )
        exhausted.players[0].position = (3, 1)
        away_from_return = mdp._single_agent_navigation_potential(exhausted)
        exhausted.players[0].position = (6, 6)
        beside_return = mdp._single_agent_navigation_potential(exhausted)
        self.assertGreater(beside_return, away_from_return)

    def test_exhausted_plate_wash_progress_is_policy_invariant_shaping(self):
        mdp = BurgerGridworld()
        state = mdp.get_start_state(1, stage="wash_required_ready")
        transition = mdp.get_state_transition(state, [Action.PROCESS])

        self.assertEqual(transition.state.sink.wash_progress, 1)
        self.assertEqual(
            transition.info["reward_breakdown"]["wash_progress"],
            0.0,
        )
        self.assertGreater(
            transition.info["reward_breakdown"]["potential"],
            0.0,
        )

    def test_navigation_shaping_penalizes_duplicate_raw_beef_pickup(self):
        mdp = BurgerGridworld()
        state = mdp.get_start_state(1, stage="rack_ready")
        state.players[0].position = (3, 1)
        mdp.validate_state(state)

        duplicate = mdp.get_state_transition(
            state, [Action.PICK_DROP]
        )

        self.assertEqual(
            duplicate.state.players[0].held_object, "raw_beef"
        )
        self.assertEqual(duplicate.state.grill.food, "raw_beef")
        self.assertLess(
            duplicate.info["reward_breakdown"]["potential"], 0.0
        )

    def test_each_plate_requires_full_wash_duration(self):
        config = BurgerConfig(total_plates=2, wash_steps=3)
        mdp = BurgerGridworld(layout((" W", "  "), ((0, 0),)), config)
        state = BurgerState(
            players=[
                BurgerPlayerState((0, 0), Direction.EAST, "dirty_plate")
            ],
            clean_plates=0,
            dirty_plates_at_return=1,
            total_plates=2,
        )
        state = mdp.get_state_transition(state, [Action.PICK_DROP]).state
        self.assertTrue(state.sink.has_dirty_plate)
        self.assertEqual(state.sink.wash_progress, 0)

        for expected in (1, 2):
            state = mdp.get_state_transition(state, [Action.PROCESS]).state
            self.assertEqual(state.sink.wash_progress, expected)
            self.assertEqual(state.clean_plates, 0)
        completed = mdp.get_state_transition(state, [Action.PROCESS]).state
        self.assertFalse(completed.sink.has_dirty_plate)
        self.assertEqual(completed.clean_plates, 0)
        self.assertEqual(
            completed.players[0].held_object, "clean_plate"
        )
        self.assertEqual(completed.dirty_plates_at_return, 1)

    def test_pick_drop_and_process_are_distinct_context_actions(self):
        mdp = BurgerGridworld(
            layout((" W", "  "), ((0, 0),)),
            BurgerConfig(total_plates=1, wash_steps=2),
        )
        state = BurgerState(
            players=[
                BurgerPlayerState(
                    (0, 0), Direction.EAST, "dirty_plate"
                )
            ],
            clean_plates=0,
            total_plates=1,
        )

        cannot_process_held_plate = mdp.get_state_transition(
            state, [Action.PROCESS]
        ).state
        self.assertEqual(
            cannot_process_held_plate.players[0].held_object,
            "dirty_plate",
        )
        self.assertFalse(cannot_process_held_plate.sink.has_dirty_plate)

        deposited = mdp.get_state_transition(
            cannot_process_held_plate, [Action.PICK_DROP]
        ).state
        self.assertTrue(deposited.sink.has_dirty_plate)
        self.assertEqual(deposited.sink.wash_progress, 0)

        cannot_wash_with_pick_drop = mdp.get_state_transition(
            deposited, [Action.PICK_DROP]
        ).state
        self.assertEqual(cannot_wash_with_pick_drop.sink.wash_progress, 0)

        processed = mdp.get_state_transition(
            cannot_wash_with_pick_drop, [Action.PROCESS]
        ).state
        self.assertEqual(processed.sink.wash_progress, 1)

    def test_two_agents_cannot_double_wash_progress(self):
        config = BurgerConfig(total_plates=1, wash_steps=3)
        mdp = BurgerGridworld(
            layout(("   ", " W ", "   "), ((0, 1), (2, 1))),
            config,
        )
        state = BurgerState(
            players=[
                BurgerPlayerState((0, 1), Direction.EAST),
                BurgerPlayerState((2, 1), Direction.WEST),
            ],
            clean_plates=0,
            total_plates=1,
            sink=SinkState(
                has_dirty_plate=True,
                wash_progress=0,
            ),
        )
        next_state = mdp.get_state_transition(
            state, [Action.PROCESS, Action.PROCESS]
        ).state
        self.assertEqual(next_state.sink.wash_progress, 1)

        handed_off = mdp.get_state_transition(
            next_state, [Action.STAY, Action.PROCESS]
        ).state
        self.assertEqual(handed_off.sink.wash_progress, 2)

        completed = mdp.get_state_transition(
            handed_off, [Action.STAY, Action.PROCESS]
        ).state
        self.assertFalse(completed.sink.has_dirty_plate)
        self.assertIsNone(completed.players[0].held_object)
        self.assertEqual(
            completed.players[1].held_object, "clean_plate"
        )

    def test_in_progress_wash_does_not_reserve_or_crash_an_agent(self):
        mdp = BurgerGridworld()
        state = BurgerState(
            players=[
                BurgerPlayerState((0, 1), Direction.NORTH),
            ],
            clean_plates=3,
            total_plates=4,
            sink=SinkState(
                has_dirty_plate=True,
                wash_progress=1,
            ),
        )

        self.assertTrue(
            mdp.context_action_available(
                state, 0, Action.PICK_DROP
            )
        )
        fetched = mdp.get_state_transition(
            state, [Action.PICK_DROP]
        ).state

        self.assertEqual(fetched.players[0].held_object, "bun")
        self.assertTrue(fetched.sink.has_dirty_plate)
        self.assertEqual(fetched.sink.wash_progress, 1)

    def test_grill_uses_environment_clock_and_fire_rewards_once(self):
        config = BurgerConfig(total_plates=1, cook_steps=3, burn_steps=2)
        mdp = BurgerGridworld(layout((" G", "  "), ((0, 0),)), config)
        state = BurgerState(
            players=[
                BurgerPlayerState((0, 0), Direction.EAST, "raw_beef")
            ],
            clean_plates=1,
            total_plates=1,
        )
        state = mdp.get_state_transition(state, [Action.PICK_DROP]).state
        self.assertEqual(state.grill.food, "raw_beef")
        self.assertEqual(state.grill.cook_ticks, 1)
        state = mdp.get_state_transition(state, [Action.STAY]).state
        ready = mdp.get_state_transition(state, [Action.STAY]).state
        self.assertEqual(ready.grill.food, "cooked_beef")

        almost_burnt = mdp.get_state_transition(ready, [Action.STAY])
        self.assertEqual(almost_burnt.state.grill.food, "cooked_beef")
        burnt = mdp.get_state_transition(almost_burnt.state, [Action.STAY])
        self.assertEqual(burnt.state.grill.food, "burnt_beef")
        self.assertEqual(burnt.info["reward_breakdown"]["fire_started"], -5.0)
        self.assertEqual(burnt.info["reward_breakdown"]["fire_active"], -0.25)
        later = mdp.get_state_transition(burnt.state, [Action.STAY])
        self.assertEqual(later.info["reward_breakdown"]["fire_started"], 0.0)
        self.assertEqual(later.info["reward_breakdown"]["fire_active"], -0.25)

    def test_fire_food_handling_is_legal_but_receives_safety_cost(self):
        config = BurgerConfig(
            total_plates=1,
            reward=BurgerRewardConfig(fire_food_handling=-2.0),
        )
        mdp = BurgerGridworld(
            layout(("B ", "  "), ((0, 1),)),
            config,
        )
        fire = BurgerState(
            players=[
                BurgerPlayerState(
                    (0, 1),
                    Direction.NORTH,
                    "clean_plate",
                )
            ],
            clean_plates=0,
            total_plates=1,
            grill=GrillState(food="burnt_beef"),
        )
        handled = mdp.get_state_transition(fire, [Action.PICK_DROP])

        self.assertEqual(
            handled.state.players[0].held_object,
            "plate_bun",
        )
        self.assertEqual(
            handled.info["reward_breakdown"]["fire_food_handling"],
            -2.0,
        )
        event = next(
            event
            for event in handled.info["events"]
            if event["type"] == "fire_food_handling"
        )
        self.assertEqual(
            event["source_event"],
            "ingredient_added_from_dispenser",
        )

        no_fire = copy.deepcopy(fire)
        no_fire.grill = GrillState()
        normal = mdp.get_state_transition(no_fire, [Action.PICK_DROP])
        self.assertEqual(
            normal.info["reward_breakdown"]["fire_food_handling"],
            0.0,
        )

    def test_default_grill_timing_and_fire_lockout(self):
        config = BurgerConfig()
        self.assertEqual(config.cook_steps, 24)
        self.assertEqual(config.burn_steps, 16)
        self.assertEqual(config.reward.fire_started, -5.0)

        mdp = BurgerGridworld(layout((" G", "  "), ((0, 0),)), config)
        state = BurgerState(
            players=[
                BurgerPlayerState((0, 0), Direction.EAST, "raw_beef")
            ],
            clean_plates=config.total_plates,
            total_plates=config.total_plates,
            grill=GrillState(
                food="burnt_beef",
                cook_ticks=config.cook_steps,
                ready_ticks=config.burn_steps,
            ),
        )
        blocked = mdp.get_state_transition(state, [Action.PICK_DROP]).state
        self.assertEqual(blocked.grill.food, "burnt_beef")
        self.assertEqual(blocked.players[0].held_object, "raw_beef")

        blocked.players[0].held_object = "extinguisher"
        blocked.extinguisher_available = False
        recovered = mdp.get_state_transition(
            blocked, [Action.PROCESS]
        )
        self.assertIsNone(recovered.state.grill.food)
        self.assertEqual(
            recovered.state.players[0].held_object,
            "extinguisher",
        )
        self.assertIn(
            "fire_extinguished",
            {event["type"] for event in recovered.info["events"]},
        )

    def test_collisions_and_swaps_cancel_joint_movement(self):
        mdp = BurgerGridworld(
            layout(("   ", "   "), ((0, 0), (2, 0))),
            BurgerConfig(total_plates=1),
        )
        state = mdp.get_standard_start_state(2)
        collided = mdp.get_state_transition(
            state, [Direction.EAST, Direction.WEST]
        ).state
        self.assertEqual(
            tuple(player.position for player in collided.players),
            ((0, 0), (2, 0)),
        )

        swap_state = BurgerState(
            players=[
                BurgerPlayerState((0, 0), Direction.EAST),
                BurgerPlayerState((1, 0), Direction.WEST),
            ],
            clean_plates=1,
            total_plates=1,
        )
        swapped = mdp.get_state_transition(
            swap_state, [Direction.EAST, Direction.WEST]
        ).state
        self.assertEqual(
            tuple(player.position for player in swapped.players),
            ((0, 0), (1, 0)),
        )

    def test_collision_does_not_freeze_unrelated_agent(self):
        mdp = BurgerGridworld(
            layout(("    ", "    "), ((0, 0), (2, 0), (0, 1))),
            BurgerConfig(total_plates=1),
        )
        transition = mdp.get_state_transition(
            mdp.get_standard_start_state(3),
            [Direction.EAST, Direction.WEST, Direction.EAST],
        )
        self.assertEqual(
            tuple(player.position for player in transition.state.players),
            ((0, 0), (2, 0), (1, 1)),
        )
        collision = next(
            event
            for event in transition.info["events"]
            if event["type"] == "collision"
        )
        self.assertEqual(collision["agents"], (0, 1))

    def test_standalone_collision_contract_is_exhaustive_and_deterministic(self):
        burger = BurgerGridworld(
            config=BurgerConfig(
                total_plates=1,
                strict_agent_obstacles=False,
            )
        )
        points = ((0, 0), (1, 0), (2, 0))
        for old_positions in permutations(points, 2):
            for new_positions in product(points, repeat=2):
                duplicate_endpoint = len(set(new_positions)) != 2
                head_on_swap = (
                    new_positions[0] == old_positions[1]
                    and new_positions[1] == old_positions[0]
                )
                self.assertEqual(
                    burger._is_transition_collision(
                        old_positions, new_positions
                    ),
                    duplicate_endpoint or head_on_swap,
                    (old_positions, new_positions),
                )

    def test_strict_agent_obstacle_blocks_following_into_vacated_cell(self):
        mdp = BurgerGridworld(
            layout(("   ", "   "), ((0, 0), (1, 0))),
            BurgerConfig(total_plates=1, strict_agent_obstacles=True),
        )
        state = mdp.get_standard_start_state(2)
        transition = mdp.get_state_transition(
            state, [Direction.EAST, Direction.EAST]
        )
        self.assertEqual(
            tuple(player.position for player in transition.state.players),
            ((0, 0), (2, 0)),
        )
        collision = next(
            event
            for event in transition.info["events"]
            if event["type"] == "collision"
        )
        self.assertEqual(collision["agents"], (0,))

    def test_plate_conservation_rejects_forged_state(self):
        mdp = BurgerGridworld()
        forged = mdp.get_standard_start_state(1)
        forged.clean_plates += 1
        with self.assertRaisesRegex(ValueError, "Plate conservation"):
            mdp.get_state_transition(forged, [Action.STAY])

    def test_random_joint_actions_preserve_all_invariants(self):
        randomizer = random.Random(20260728)
        mdp = BurgerGridworld(
            config=BurgerConfig(horizon=1200, total_plates=4)
        )
        env = BurgerEnv(mdp=mdp, num_players=4)
        previous_digest = None
        for _ in range(1000):
            transition = env.step(
                [randomizer.choice(Action.ALL_ACTIONS) for _ in range(4)]
            )
            mdp.validate_state(transition.state)
            self.assertEqual(len(transition.info["state_digest"]), 64)
            self.assertNotEqual(transition.info["state_digest"], previous_digest)
            previous_digest = transition.info["state_digest"]


if __name__ == "__main__":
    unittest.main()
