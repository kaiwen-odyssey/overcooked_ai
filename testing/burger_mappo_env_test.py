import copy
import unittest

import numpy as np

from burger_marl.actions import Action, Direction
from burger_marl.mappo_env import (
    BurgerMAPPOEnv,
    LOCAL_CHANNELS,
    MAX_AGENTS,
    NUM_GLOBAL_CHANNELS,
    NUM_LOCAL_CHANNELS,
    TERRAIN_CHANNEL,
)
from burger_marl.env import COUNTER, RETURN, SERVE, TRASH


class TestBurgerMAPPOContract(unittest.TestCase):
    def test_standalone_action_contract_has_seven_explicit_actions(self):
        self.assertEqual(Action.NUM_ACTIONS, 7)
        self.assertEqual(
            Action.INDEX_TO_ACTION,
            [
                Direction.NORTH,
                Direction.SOUTH,
                Direction.EAST,
                Direction.WEST,
                Action.STAY,
                Action.PICK_DROP,
                Action.PROCESS,
            ],
        )

    def test_fixed_shapes_for_one_to_four_active_agents(self):
        self.assertEqual(NUM_LOCAL_CHANNELS, 26)
        self.assertEqual(NUM_GLOBAL_CHANNELS, 44)
        for num_players in range(1, MAX_AGENTS + 1):
            env = BurgerMAPPOEnv(num_players=num_players)
            observations, shared, available = env.reset(
                seed=20260728
            )
            self.assertEqual(
                observations.shape,
                (MAX_AGENTS, NUM_LOCAL_CHANNELS, 9, 9),
            )
            self.assertEqual(shared.shape[0], MAX_AGENTS)
            self.assertEqual(
                available.shape,
                (MAX_AGENTS, Action.NUM_ACTIONS),
            )
            self.assertTrue(np.isfinite(observations).all())
            self.assertTrue(np.isfinite(shared).all())
            for index in range(num_players):
                self.assertTrue(
                    env.observation_space[index].contains(
                        observations[index]
                    )
                )
                self.assertTrue(
                    env.share_observation_space[index].contains(
                        shared[index]
                    )
                )
            self.assertTrue(
                np.all(observations[num_players:] == 0)
            )
            self.assertTrue(np.all(shared[num_players:] == 0))
            self.assertTrue(
                np.all(env.active_masks()[:num_players] == 1)
            )
            self.assertTrue(
                np.all(env.active_masks()[num_players:] == 0)
            )

    def test_layout_has_no_virtual_assembly_inventory(self):
        env = BurgerMAPPOEnv(num_players=1)
        self.assertEqual(env.mdp.layout.terrain_at((5, 7)), COUNTER)
        self.assertFalse(hasattr(env.state, "assembly"))

    def test_action_mask_exposes_only_valid_visible_context_actions(self):
        env = BurgerMAPPOEnv(num_players=1)
        stay = Action.ACTION_TO_INDEX[Action.STAY]
        pick_drop = Action.ACTION_TO_INDEX[Action.PICK_DROP]
        process = Action.ACTION_TO_INDEX[Action.PROCESS]
        available = env.available_actions()

        self.assertEqual(available[0, pick_drop], 1.0)
        self.assertEqual(available[0, process], 0.0)
        self.assertEqual(available[0, :stay + 1].sum(), 3.0)
        self.assertEqual(available[0].sum(), 4.0)

        transition = env.step(
            [pick_drop, stay, stay, stay]
        )
        next_available = transition[-1]
        self.assertEqual(next_available[0, pick_drop], 0.0)
        self.assertEqual(next_available[0, process], 0.0)
        self.assertEqual(next_available[0].sum(), 3.0)

    def test_inactive_slots_can_only_submit_stay(self):
        env = BurgerMAPPOEnv(num_players=2)
        stay = Action.ACTION_TO_INDEX[Action.STAY]
        interact = Action.ACTION_TO_INDEX[Action.PICK_DROP]
        with self.assertRaisesRegex(ValueError, "Inactive"):
            env.step([stay, stay, interact, stay])

    def test_step_returns_shared_team_reward_and_agent_masks(self):
        env = BurgerMAPPOEnv(num_players=3)
        stay = Action.ACTION_TO_INDEX[Action.STAY]
        (
            observations,
            shared,
            rewards,
            dones,
            infos,
            available,
        ) = env.step([stay] * MAX_AGENTS)
        self.assertEqual(observations.shape[0], MAX_AGENTS)
        self.assertEqual(shared.shape[0], MAX_AGENTS)
        self.assertEqual(rewards.shape, (MAX_AGENTS, 1))
        self.assertAlmostEqual(rewards[0, 0], rewards[1, 0])
        self.assertAlmostEqual(rewards[1, 0], rewards[2, 0])
        self.assertEqual(rewards[3, 0], 0.0)
        self.assertFalse(dones[:3].any())
        self.assertTrue(dones[3])
        self.assertTrue(all(info["active"] for info in infos[:3]))
        self.assertFalse(infos[3]["active"])
        self.assertEqual(
            available[3, stay],
            1.0,
        )
        self.assertEqual(available[3].sum(), 1.0)

    def test_actor_observation_does_not_leak_distant_global_state(self):
        env = BurgerMAPPOEnv(
            num_players=1,
            observation_radius=1,
            occlusion=True,
        )
        visible_state = env.state
        hidden_state = copy.deepcopy(visible_state)
        hidden_state.counter_objects[(7, 3)] = "bun"

        local_before = env._local_observation(visible_state, 0)
        local_after = env._local_observation(hidden_state, 0)
        global_before = env._global_state(visible_state)
        global_after = env._global_state(hidden_state)
        np.testing.assert_array_equal(local_before, local_after)
        self.assertFalse(np.array_equal(global_before, global_after))

    def test_actor_observation_identifies_visible_trash_station(self):
        env = BurgerMAPPOEnv(num_players=1)
        state = env.state
        state.players[0].position = (7, 1)
        state.players[0].orientation = Direction.NORTH

        observation = env._local_observation(state, 0)
        center = env.observation_radius

        self.assertEqual(
            observation[TERRAIN_CHANNEL[TRASH], center - 1, center],
            1.0,
        )

    def test_actor_distinguishes_delivery_from_dish_return_hatch(self):
        env = BurgerMAPPOEnv(num_players=1)
        state = env.state
        state.players[0].position = (6, 6)
        state.players[0].orientation = Direction.SOUTH

        observation = env._local_observation(state, 0)
        center = env.observation_radius

        self.assertEqual(
            observation[TERRAIN_CHANNEL[RETURN], center + 1, center],
            1.0,
        )
        self.assertEqual(
            observation[TERRAIN_CHANNEL[SERVE], center + 1, center],
            0.0,
        )

    def test_actor_observes_whether_unique_extinguisher_is_at_station(self):
        env = BurgerMAPPOEnv(num_players=1)
        available = env.state
        available.players[0].position = (6, 1)
        unavailable = copy.deepcopy(available)
        unavailable.extinguisher_available = False

        with_tool = env._local_observation(available, 0)
        without_tool = env._local_observation(unavailable, 0)
        center = env.observation_radius

        self.assertEqual(
            with_tool[
                LOCAL_CHANNELS["extinguisher"],
                center - 1,
                center,
            ],
            1.0,
        )
        self.assertEqual(
            without_tool[
                LOCAL_CHANNELS["extinguisher"],
                center - 1,
                center,
            ],
            0.0,
        )

    def test_actor_local_observation_is_world_aligned_not_orientation_aligned(self):
        env = BurgerMAPPOEnv(num_players=1)
        original = env.state
        turned = copy.deepcopy(original)
        turned.players[0].orientation = Direction.EAST

        np.testing.assert_array_equal(
            env._local_observation(original, 0),
            env._local_observation(turned, 0),
        )
        self.assertFalse(
            np.array_equal(
                env._global_state(original),
                env._global_state(turned),
            )
        )

    def test_critic_state_distinguishes_markov_relevant_hidden_state(self):
        env = BurgerMAPPOEnv(num_players=1)
        original = env.state
        turned = copy.deepcopy(original)
        turned.players[0].orientation = Direction.EAST
        self.assertFalse(
            np.array_equal(
                env._global_state(original),
                env._global_state(turned),
            )
        )

        plate_in_flight = copy.deepcopy(original)
        plate_in_flight.clean_plates -= 1
        plate_in_flight.pending_plate_returns = [4]
        self.assertFalse(
            np.array_equal(
                env._global_state(original),
                env._global_state(plate_in_flight),
            )
        )

    def test_state_property_cannot_mutate_authoritative_environment(self):
        env = BurgerMAPPOEnv(num_players=1)
        external = env.state
        external.clean_plates = 999
        self.assertNotEqual(env.state.clean_plates, 999)

    def test_same_seed_and_actions_are_deterministic(self):
        stay = Action.ACTION_TO_INDEX[Action.STAY]
        east = Action.ACTION_TO_INDEX[(1, 0)]
        actions = [
            [east, stay, stay, stay],
            [stay, stay, stay, stay],
            [east, stay, stay, stay],
        ]
        left = BurgerMAPPOEnv(num_players=1)
        right = BurgerMAPPOEnv(num_players=1)
        left.reset(seed=17)
        right.reset(seed=17)
        for joint_action in actions:
            left_step = left.step(joint_action)
            right_step = right.step(joint_action)
            np.testing.assert_array_equal(
                left_step[0], right_step[0]
            )
            np.testing.assert_array_equal(
                left_step[1], right_step[1]
            )
            np.testing.assert_array_equal(
                left_step[2], right_step[2]
            )
            self.assertEqual(
                left_step[4][0]["state_digest"],
                right_step[4][0]["state_digest"],
            )

    def test_repeated_invalid_context_actions_cannot_farm_reward(self):
        env = BurgerMAPPOEnv(num_players=1)
        stay = Action.ACTION_TO_INDEX[Action.STAY]
        pick_drop = Action.ACTION_TO_INDEX[Action.PICK_DROP]
        process = Action.ACTION_TO_INDEX[Action.PROCESS]
        env.step([pick_drop, stay, stay, stay])
        rewards = []
        for action in [pick_drop, process] * 10:
            transition = env.step(
                [action, stay, stay, stay]
            )
            rewards.append(float(transition[2][0, 0]))
            event_types = {
                event["type"]
                for event in transition[4][0]["events"]
            }
            self.assertNotIn("correct_delivery", event_types)
        self.assertLessEqual(max(rewards), 0.0)

    def test_rejects_non_integer_and_wrong_sized_actions(self):
        env = BurgerMAPPOEnv(num_players=1)
        stay = Action.ACTION_TO_INDEX[Action.STAY]
        with self.assertRaisesRegex(ValueError, "four fixed"):
            env.step([stay])
        with self.assertRaisesRegex(ValueError, "integers"):
            env.step([0.0, stay, stay, stay])


if __name__ == "__main__":
    unittest.main()
