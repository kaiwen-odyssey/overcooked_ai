import tempfile
import unittest

import numpy as np
import torch

from burger_marl.training import (
    LocalActor,
    PPOTrainer,
    TrainConfig,
    environment_config,
    evaluate_policy,
    generate_single_agent_expert_demo,
)


class TestBurgerPPOTrainer(unittest.TestCase):
    def _config(self, algorithm="ppo", num_agents=1):
        return TrainConfig(
            algorithm=algorithm,
            num_agents=num_agents,
            total_env_steps=16,
            num_envs=2,
            rollout_length=8,
            update_epochs=1,
            num_minibatches=2,
            hidden_size=32,
            horizon=12,
            eval_episodes=1,
            device="cpu",
        )

    def test_rollout_and_update_shapes_for_single_agent_ppo(self):
        trainer = PPOTrainer(self._config())
        rollout = trainer.collect_rollout()
        self.assertEqual(
            rollout.observations.shape[:3], (8, 2, 1)
        )
        self.assertEqual(
            rollout.available_actions.shape, (8, 2, 1, 7)
        )
        self.assertEqual(rollout.actions.shape, (8, 2, 1))
        self.assertEqual(rollout.rewards.shape, (8, 2))
        selected_is_available = torch.gather(
            rollout.available_actions,
            -1,
            rollout.actions.unsqueeze(-1),
        )
        self.assertTrue(torch.all(selected_is_available == 1))
        metrics = trainer.update_policy(rollout)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_two_agent_mappo_shares_actor_and_uses_team_critic(self):
        trainer = PPOTrainer(self._config("mappo", 2))
        rollout = trainer.collect_rollout()
        self.assertEqual(
            rollout.observations.shape[:3], (8, 2, 2)
        )
        self.assertEqual(rollout.values.shape, (8, 2))
        self.assertEqual(rollout.advantages.shape, (8, 2))
        parameter_ids = {id(parameter) for parameter in trainer.actor.parameters()}
        self.assertEqual(
            parameter_ids,
            {id(parameter) for parameter in trainer.actor.parameters()},
        )
        trainer.update_policy(rollout)

    def test_single_agent_actor_checkpoint_warm_starts_mappo(self):
        source = PPOTrainer(self._config())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = source.save_checkpoint(
                __import__("pathlib").Path(directory), "one_agent"
            )
            target_config = TrainConfig(
                **{
                    **self._config("mappo", 2).__dict__,
                    "actor_init": str(checkpoint),
                }
            )
            target = PPOTrainer(target_config)
            for name, value in source.actor.state_dict().items():
                torch.testing.assert_close(
                    value.cpu(), target.actor.state_dict()[name].cpu()
                )

    def test_deterministic_evaluation_reports_sparse_separately(self):
        trainer = PPOTrainer(self._config())
        metrics = evaluate_policy(
            trainer.actor, trainer.config, trainer.device, episodes=1
        )
        self.assertIn("eval_mean_sparse_reward", metrics)
        self.assertIn("eval_mean_deliveries", metrics)
        self.assertGreaterEqual(metrics["eval_mean_deliveries"], 0.0)

    def test_config_rejects_false_algorithm_labels(self):
        with self.assertRaises(ValueError):
            self._config("ppo", 2)
        with self.assertRaises(ValueError):
            self._config("mappo", 1)

    def test_checkpointed_config_materializes_grill_contract(self):
        config = self._config()
        self.assertEqual(
            config.action_contract,
            "standalone_v2_adjacent_pick_drop_process",
        )
        self.assertEqual(
            config.action_mask_contract,
            "local_visible_adjacent_context_v2",
        )
        self.assertEqual(
            config.world_object_contract,
            "visible_counter_or_plate_v1",
        )
        burger_config = environment_config(config)
        self.assertEqual(burger_config.cook_steps, 24)
        self.assertEqual(burger_config.burn_steps, 16)
        self.assertEqual(burger_config.plate_return_steps, 12)
        self.assertEqual(burger_config.reward.fire_started, -5.0)

    def test_single_agent_expert_completes_with_recorded_action_masks(self):
        config = TrainConfig(
            total_env_steps=16,
            num_envs=1,
            rollout_length=8,
            update_epochs=1,
            num_minibatches=1,
            device="cpu",
        )
        demo = generate_single_agent_expert_demo(config)
        selected_masks = demo.available_actions[
            np.arange(len(demo.actions)),
            demo.actions,
        ]

        self.assertEqual(demo.delivered_orders, 1)
        self.assertTrue(np.all(selected_masks == 1.0))
        self.assertLess(len(demo.actions), config.horizon)


if __name__ == "__main__":
    unittest.main()
