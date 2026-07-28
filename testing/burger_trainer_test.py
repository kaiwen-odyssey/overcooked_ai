import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from burger_marl.actions import Action
from burger_marl.training import (
    BurgerVectorEnv,
    PPOTrainer,
    TrainConfig,
    _masked_logits,
    audit_training_contract,
    environment_config,
    evaluate_policy,
    generate_single_agent_expert_demo,
    parse_args,
    train,
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
        before = {
            name: value.detach().clone()
            for name, value in trainer.actor.state_dict().items()
        }
        metrics = trainer.update_policy(rollout)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertGreater(metrics["grad_norm"], 0.0)
        self.assertTrue(
            any(
                not torch.equal(before[name], value)
                for name, value in trainer.actor.state_dict().items()
            )
        )
        self.assertEqual(
            sum(rollout.action_counts),
            8 * 2,
        )
        self.assertEqual(rollout.joint_steps, 8 * 2)
        self.assertGreaterEqual(rollout.mean_legal_actions, 1.0)
        self.assertAlmostEqual(
            sum(rollout.reward_component_totals.values()),
            float(rollout.rewards.sum()),
            places=4,
        )
        self.assertIn("rollout_all_agents_stay_ratio", metrics)
        self.assertIn("action_fraction_process", metrics)
        self.assertIn("reward_component_potential", metrics)

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
                Path(directory), "one_agent"
            )
            target_config = TrainConfig(
                **{
                    **self._config("mappo", 2).__dict__,
                    "actor_init": str(checkpoint),
                }
            )
            target = PPOTrainer(target_config)
            for name, value in source.actor.state_dict().items():
                if name == "agent_embedding.weight":
                    continue
                torch.testing.assert_close(
                    value.cpu(), target.actor.state_dict()[name].cpu()
                )
            source_embedding = (
                source.actor.agent_embedding.weight[0].detach().cpu()
            )
            for embedding in target.actor.agent_embedding.weight:
                torch.testing.assert_close(
                    embedding.detach().cpu(), source_embedding
                )

    def test_mappo_actor_init_rejects_checkpoint_without_agent_metadata(self):
        source = PPOTrainer(self._config())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "actor_only.pt"
            torch.save({"actor": source.actor.state_dict()}, checkpoint)

            with self.assertRaisesRegex(
                ValueError, "train_config.num_agents"
            ):
                PPOTrainer(
                    TrainConfig(
                        **{
                            **self._config("mappo", 2).__dict__,
                            "actor_init": str(checkpoint),
                        }
                    )
                )

    def test_deterministic_evaluation_reports_sparse_separately(self):
        trainer = PPOTrainer(self._config())
        metrics = evaluate_policy(
            trainer.actor, trainer.config, trainer.device, episodes=1
        )
        self.assertIn("eval_mean_sparse_reward", metrics)
        self.assertIn("eval_mean_deliveries", metrics)
        self.assertIn("eval_stay_action_ratio", metrics)
        self.assertIn("eval_all_agents_stay_ratio", metrics)
        self.assertIn("eval_all_agents_noop_ratio", metrics)
        self.assertIn("eval_action_coverage", metrics)
        self.assertGreaterEqual(metrics["eval_mean_deliveries"], 0.0)
        self.assertGreaterEqual(metrics["eval_action_coverage"], 0.0)
        self.assertLessEqual(metrics["eval_action_coverage"], 1.0)

    def test_config_rejects_false_algorithm_labels(self):
        with self.assertRaises(ValueError):
            self._config("ppo", 2)
        with self.assertRaises(ValueError):
            self._config("mappo", 1)
        with self.assertRaisesRegex(ValueError, "one-agent PPO"):
            TrainConfig(
                algorithm="mappo",
                num_agents=2,
                bc_pretrain_steps=1,
            )

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
        self.assertEqual(burger_config.wash_steps, 10)
        self.assertEqual(burger_config.plate_return_steps, 12)
        self.assertEqual(burger_config.reward.correct_delivery, 20.0)
        self.assertEqual(burger_config.reward.fire_started, -5.0)
        self.assertEqual(burger_config.reward.collision, -0.05)
        self.assertEqual(burger_config.reward.time_step, -0.01)
        self.assertEqual(
            burger_config.reward.gamma, config.gamma
        )

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

    def test_training_preflight_checks_every_action_and_reward_source(self):
        config = self._config()
        report = audit_training_contract(config)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["action_count"], 7)
        self.assertEqual(
            set(report["actions"]),
            {
                "north",
                "south",
                "east",
                "west",
                "stay",
                "pick_drop",
                "process",
            },
        )
        self.assertEqual(
            set(report["reward_components"]),
            {
                "correct_delivery",
                "fire_started",
                "collision",
                "time_step",
                "potential_scale",
                "potential_gamma",
            },
        )
        self.assertTrue(report["active_masks_never_empty"])
        self.assertTrue(report["vector_masks_refresh_after_step"])
        self.assertTrue(
            report["invalid_interactions_have_no_positive_reward"]
        )
        self.assertTrue(report["potential_gamma_matches_ppo_gamma"])
        self.assertTrue(report["wash_progress_does_not_reserve_agent"])
        self.assertTrue(report["sink_process_mask_has_no_hidden_owner"])
        self.assertTrue(
            report["unrelated_agents_continue_after_local_collision"]
        )

    def test_vector_environment_refreshes_masks_after_every_step(self):
        config = self._config()
        env = BurgerVectorEnv(
            num_envs=1,
            num_agents=1,
            burger_config=environment_config(config),
        )
        pick_drop = Action.ACTION_TO_INDEX[Action.PICK_DROP]
        stay = Action.ACTION_TO_INDEX[Action.STAY]
        self.assertEqual(
            env.available_actions[0, 0, pick_drop], 1.0
        )

        actions = np.full((1, 4), stay, dtype=np.int64)
        actions[0, 0] = pick_drop
        env.step(actions)

        self.assertEqual(
            env.available_actions[0, 0, pick_drop], 0.0
        )
        self.assertEqual(env.available_actions[0, 0, stay], 1.0)

    def test_action_freeze_guard_aborts_before_another_update(self):
        base = self._config()
        config = TrainConfig(
            **{
                **base.__dict__,
                "action_freeze_patience_updates": 1,
            }
        )
        trainer = PPOTrainer(config)
        rollout = trainer.collect_rollout()
        stay_index = Action.ACTION_TO_INDEX[Action.STAY]
        counts = [0] * Action.NUM_ACTIONS
        counts[stay_index] = sum(rollout.action_counts)
        frozen = replace(
            rollout,
            action_counts=tuple(counts),
            all_agents_stay_steps=rollout.joint_steps,
            all_agents_noop_steps=rollout.joint_steps,
        )

        with self.assertRaisesRegex(RuntimeError, "Action-freeze gate"):
            trainer.update_policy(frozen)

    def test_masked_logits_reject_all_zero_or_non_binary_masks(self):
        logits = torch.zeros((2, Action.NUM_ACTIONS))
        with self.assertRaisesRegex(ValueError, "at least one"):
            _masked_logits(logits, torch.zeros_like(logits))

        non_binary = torch.ones_like(logits)
        non_binary[0, 0] = 0.5
        with self.assertRaisesRegex(ValueError, "binary"):
            _masked_logits(logits, non_binary)

    def test_nonfinite_gradient_aborts_before_optimizer_step(self):
        trainer = PPOTrainer(self._config())
        rollout = trainer.collect_rollout()
        parameters_before = {
            name: parameter.detach().clone()
            for name, parameter in (
                list(trainer.actor.named_parameters())
                + [
                    ("critic." + name, parameter)
                    for name, parameter in trainer.critic.named_parameters()
                ]
            )
        }
        poisoned = replace(
            rollout,
            advantages=torch.full_like(
                rollout.advantages, float("nan")
            ),
        )

        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            trainer.update_policy(poisoned)
        parameters_after = dict(trainer.actor.named_parameters()) | {
            "critic." + name: parameter
            for name, parameter in trainer.critic.named_parameters()
        }
        for name, parameter in parameters_after.items():
            torch.testing.assert_close(
                parameter.detach(), parameters_before[name]
            )

    def test_train_runs_requested_behavior_cloning_and_saves_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            config = TrainConfig(
                total_env_steps=1,
                num_envs=1,
                rollout_length=1,
                update_epochs=1,
                num_minibatches=1,
                hidden_size=16,
                horizon=60,
                eval_episodes=1,
                eval_interval_updates=1,
                checkpoint_interval_updates=1,
                device="cpu",
                output_dir=directory,
                bc_pretrain_steps=1,
                bc_batch_size=8,
            )

            checkpoint = train(config)
            run_dir = checkpoint.parent
            pretraining = json.loads(
                (run_dir / "pretraining.json").read_text(
                    encoding="utf-8"
                )
            )
            summary = json.loads(
                (run_dir / "summary.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(pretraining["bc_steps"], 1.0)
            self.assertEqual(summary["pretraining"]["bc_steps"], 1.0)
            self.assertTrue((run_dir / "pretrained.pt").exists())

    def test_train_refuses_to_overwrite_an_existing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = (
                Path(directory) / "ppo_1agent_v2_seed20260728"
            )
            run_dir.mkdir()
            sentinel = run_dir / "keep.txt"
            sentinel.write_text("original", encoding="utf-8")
            config = TrainConfig(
                total_env_steps=1,
                num_envs=1,
                rollout_length=1,
                update_epochs=1,
                num_minibatches=1,
                output_dir=directory,
                device="cpu",
            )

            with self.assertRaisesRegex(
                FileExistsError, "Refusing to overwrite"
            ):
                train(config)
            self.assertEqual(
                sentinel.read_text(encoding="utf-8"), "original"
            )

    def test_cli_exposes_behavior_cloning_controls(self):
        config = parse_args(
            [
                "--algorithm",
                "ppo",
                "--num-agents",
                "1",
                "--bc-pretrain-steps",
                "7",
                "--bc-batch-size",
                "16",
                "--bc-learning-rate",
                "0.002",
                "--bc-aux-coef",
                "0.1",
            ]
        )

        self.assertEqual(config.bc_pretrain_steps, 7)
        self.assertEqual(config.bc_batch_size, 16)
        self.assertEqual(config.bc_learning_rate, 0.002)
        self.assertEqual(config.bc_aux_coef, 0.1)


if __name__ == "__main__":
    unittest.main()
