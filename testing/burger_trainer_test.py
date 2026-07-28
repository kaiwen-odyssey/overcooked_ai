import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from burger_marl.actions import Action
from burger_marl.env import EXTINGUISHER, SINK
from burger_marl.mappo_env import LOCAL_CHANNELS, TERRAIN_CHANNEL
from burger_marl.training import (
    BurgerVectorEnv,
    PPOTrainer,
    TrainConfig,
    _masked_logits,
    audit_training_contract,
    environment_config,
    evaluate_configured_stages,
    evaluate_policy,
    generate_single_agent_expert_demo,
    load_actor_state_dict_compatible,
    parse_args,
    train,
)


class TestBurgerPPOTrainer(unittest.TestCase):
    def test_emergency_residual_is_locally_gated(self):
        trainer = PPOTrainer(self._config())
        actor = trainer.actor
        shape = actor.encoder[0].weight.shape[1]
        side = trainer.vector_env.observations.shape[-1]
        normal = torch.zeros((1, shape, side, side))
        agent_ids = torch.zeros(1, dtype=torch.long)

        with torch.no_grad():
            actor.emergency_policy[-1].bias.fill_(1.0)
            actor.cleanup_policy[-1].bias.fill_(2.0)
            actor.suppression_policy[-1].bias.fill_(3.0)
            actor.dish_policy[-1].bias.fill_(4.0)
            actor.assembly_policy[-1].bias.fill_(5.0)
            actor.workflow_policy[-1].bias.fill_(6.0)
            normal_logits = actor(normal, agent_ids)
            normal_base = actor.policy(
                torch.cat(
                    (
                        actor.encoder(normal),
                        actor.agent_embedding(agent_ids),
                    ),
                    dim=-1,
                )
            )
            fire = normal.clone()
            fire[
                0,
                LOCAL_CHANNELS["fire_alarm"],
            ] = 1.0
            fire_logits = actor(fire, agent_ids)
            fire_base = actor.policy(
                torch.cat(
                    (
                        actor.encoder(fire),
                        actor.agent_embedding(agent_ids),
                    ),
                    dim=-1,
                )
            )
            visible_extinguisher = normal.clone()
            visible_extinguisher[
                0,
                LOCAL_CHANNELS["extinguisher"],
                0,
                0,
            ] = 1.0
            visible_extinguisher[
                0,
                TERRAIN_CHANNEL[EXTINGUISHER],
                0,
                0,
            ] = 1.0
            visible_extinguisher_logits = actor(
                visible_extinguisher, agent_ids
            )
            visible_extinguisher_base = actor.policy(
                torch.cat(
                    (
                        actor.encoder(visible_extinguisher),
                        actor.agent_embedding(agent_ids),
                    ),
                    dim=-1,
                )
            )
            held_extinguisher = normal.clone()
            held_extinguisher[
                0,
                LOCAL_CHANNELS["extinguisher"],
                side // 2,
                side // 2,
            ] = 1.0
            held_extinguisher_logits = actor(
                held_extinguisher, agent_ids
            )
            held_extinguisher_base = actor.policy(
                torch.cat(
                    (
                        actor.encoder(held_extinguisher),
                        actor.agent_embedding(agent_ids),
                    ),
                    dim=-1,
                )
            )
            misplaced_extinguisher = normal.clone()
            misplaced_extinguisher[
                0,
                LOCAL_CHANNELS["extinguisher"],
                0,
                0,
            ] = 1.0
            misplaced_extinguisher_logits = actor(
                misplaced_extinguisher, agent_ids
            )
            misplaced_extinguisher_base = actor.policy(
                torch.cat(
                    (
                        actor.encoder(misplaced_extinguisher),
                        actor.agent_embedding(agent_ids),
                    ),
                    dim=-1,
                )
            )
            fire_while_holding = fire.clone()
            fire_while_holding[
                0,
                LOCAL_CHANNELS["extinguisher"],
                side // 2,
                side // 2,
            ] = 1.0
            fire_while_holding_logits = actor(
                fire_while_holding, agent_ids
            )
            fire_while_holding_base = actor.policy(
                torch.cat(
                    (
                        actor.encoder(fire_while_holding),
                        actor.agent_embedding(agent_ids),
                    ),
                    dim=-1,
                )
            )
            plate_shortage = normal.clone()
            plate_shortage[
                0,
                LOCAL_CHANNELS["plate_shortage"],
            ] = 1.0
            shortage_without_dirty_logits = actor(
                plate_shortage, agent_ids
            )
            shortage_without_dirty_base = actor.policy(
                torch.cat(
                    (
                        actor.encoder(plate_shortage),
                        actor.agent_embedding(agent_ids),
                    ),
                    dim=-1,
                )
            )
            plate_shortage_with_clean_plate = plate_shortage.clone()
            plate_shortage_with_clean_plate[
                0,
                LOCAL_CHANNELS["plate"],
                side // 2,
                side // 2,
            ] = 1.0
            plate_shortage_with_clean_plate_logits = actor(
                plate_shortage_with_clean_plate, agent_ids
            )
            plate_shortage_with_clean_plate_base = actor.policy(
                torch.cat(
                    (
                        actor.encoder(plate_shortage_with_clean_plate),
                        actor.agent_embedding(agent_ids),
                    ),
                    dim=-1,
                )
            )
            plate_shortage[
                0,
                LOCAL_CHANNELS["dirty_plate"],
                side // 2,
                side // 2,
            ] = 1.0
            plate_shortage_logits = actor(plate_shortage, agent_ids)
            plate_shortage_base = actor.policy(
                torch.cat(
                    (
                        actor.encoder(plate_shortage),
                        actor.agent_embedding(agent_ids),
                    ),
                    dim=-1,
                )
            )
            dirty_on_counter_with_clean_plate = (
                plate_shortage_with_clean_plate.clone()
            )
            dirty_on_counter_with_clean_plate[
                0,
                LOCAL_CHANNELS["dirty_plate"],
                0,
                0,
            ] = 1.0
            dirty_on_counter_with_clean_plate_logits = actor(
                dirty_on_counter_with_clean_plate, agent_ids
            )
            dirty_on_counter_with_clean_plate_base = actor.policy(
                torch.cat(
                    (
                        actor.encoder(
                            dirty_on_counter_with_clean_plate
                        ),
                        actor.agent_embedding(agent_ids),
                    ),
                    dim=-1,
                )
            )
            dirty_in_sink = shortage_without_dirty_logits.new_zeros(
                normal.shape
            )
            dirty_in_sink[
                0,
                LOCAL_CHANNELS["plate_shortage"],
            ] = 1.0
            dirty_in_sink[
                0,
                LOCAL_CHANNELS["dirty_plate"],
                0,
                0,
            ] = 1.0
            dirty_in_sink[
                0,
                TERRAIN_CHANNEL[SINK],
                0,
                0,
            ] = 1.0
            dirty_in_sink_logits = actor(dirty_in_sink, agent_ids)
            dirty_in_sink_base = actor.policy(
                torch.cat(
                    (
                        actor.encoder(dirty_in_sink),
                        actor.agent_embedding(agent_ids),
                    ),
                    dim=-1,
                )
            )
            held_bun_lettuce_plate = (
                plate_shortage_with_clean_plate.clone()
            )
            held_bun_lettuce_plate[
                0,
                LOCAL_CHANNELS["bun"],
                side // 2,
                side // 2,
            ] = 1.0
            held_bun_lettuce_plate[
                0,
                LOCAL_CHANNELS["lettuce"],
                side // 2,
                side // 2,
            ] = 1.0
            held_bun_lettuce_logits = actor(
                held_bun_lettuce_plate, agent_ids
            )
            held_bun_lettuce_base = actor.policy(
                torch.cat(
                    (
                        actor.encoder(held_bun_lettuce_plate),
                        actor.agent_embedding(agent_ids),
                    ),
                    dim=-1,
                )
            )

        torch.testing.assert_close(normal_logits, normal_base)
        torch.testing.assert_close(
            fire_logits, fire_base + torch.ones_like(fire_base)
        )
        torch.testing.assert_close(
            visible_extinguisher_logits, visible_extinguisher_base
        )
        torch.testing.assert_close(
            held_extinguisher_logits,
            held_extinguisher_base
            + 2.0 * torch.ones_like(held_extinguisher_base),
        )
        torch.testing.assert_close(
            misplaced_extinguisher_logits,
            misplaced_extinguisher_base
            + 2.0 * torch.ones_like(misplaced_extinguisher_base),
        )
        torch.testing.assert_close(
            fire_while_holding_logits,
            fire_while_holding_base
            + 3.0 * torch.ones_like(fire_while_holding_base),
        )
        torch.testing.assert_close(
            shortage_without_dirty_logits,
            shortage_without_dirty_base,
        )
        torch.testing.assert_close(
            plate_shortage_with_clean_plate_logits,
            plate_shortage_with_clean_plate_base
            + 5.0 * torch.ones_like(
                plate_shortage_with_clean_plate_base
            ),
        )
        torch.testing.assert_close(
            plate_shortage_logits,
            plate_shortage_base
            + 4.0 * torch.ones_like(plate_shortage_base),
        )
        torch.testing.assert_close(
            dirty_on_counter_with_clean_plate_logits,
            dirty_on_counter_with_clean_plate_base
            + 5.0 * torch.ones_like(
                dirty_on_counter_with_clean_plate_base
            ),
        )
        torch.testing.assert_close(
            dirty_in_sink_logits,
            dirty_in_sink_base
            + 4.0 * torch.ones_like(dirty_in_sink_base),
        )
        torch.testing.assert_close(
            held_bun_lettuce_logits,
            held_bun_lettuce_base
            + 11.0 * torch.ones_like(held_bun_lettuce_base),
        )

    def test_dish_training_freezes_standard_and_fire_actors(self):
        config = TrainConfig(
            **{
                **self._config().__dict__,
                "training_start_stage": "dirty_plate_carry_ready",
                "training_start_mix_stage": "plate_exhausted_ready",
                "freeze_base_actor_for_dish": True,
                "freeze_fire_actors_for_dish": True,
            }
        )
        trainer = PPOTrainer(config)

        frozen_modules = (
            trainer.actor.encoder,
            trainer.actor.agent_embedding,
            trainer.actor.policy,
            trainer.actor.emergency_encoder,
            trainer.actor.emergency_policy,
            trainer.actor.cleanup_encoder,
            trainer.actor.cleanup_policy,
            trainer.actor.suppression_encoder,
            trainer.actor.suppression_policy,
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for module in frozen_modules
                for parameter in module.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for module in (
                trainer.actor.dish_encoder,
                trainer.actor.dish_policy,
                trainer.actor.assembly_encoder,
                trainer.actor.assembly_policy,
            )
                for parameter in module.parameters()
            )
        )

    def test_assembly_training_can_preserve_dish_actor(self):
        config = TrainConfig(
            **{
                **self._config().__dict__,
                "training_start_stage": "dirty_plate_carry_ready",
                "freeze_base_actor_for_dish": True,
                "freeze_fire_actors_for_dish": True,
                "freeze_dish_actor_for_assembly": True,
            }
        )
        trainer = PPOTrainer(config)

        self.assertTrue(
            all(
                not parameter.requires_grad
                for module in (
                    trainer.actor.dish_encoder,
                    trainer.actor.dish_policy,
                )
                for parameter in module.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for module in (
                    trainer.actor.assembly_encoder,
                    trainer.actor.assembly_policy,
                )
                for parameter in module.parameters()
            )
        )

    def test_workflow_training_can_preserve_broad_assembly_actor(self):
        config = TrainConfig(
            **{
                **self._config().__dict__,
                "training_start_stage": "dirty_plate_carry_ready",
                "freeze_base_actor_for_dish": True,
                "freeze_fire_actors_for_dish": True,
                "freeze_dish_actor_for_assembly": True,
                "freeze_assembly_actor_for_workflow": True,
            }
        )
        trainer = PPOTrainer(config)

        self.assertTrue(
            all(
                not parameter.requires_grad
                for module in (
                    trainer.actor.assembly_encoder,
                    trainer.actor.assembly_policy,
                )
                for parameter in module.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for module in (
                    trainer.actor.workflow_encoder,
                    trainer.actor.workflow_policy,
                )
                for parameter in module.parameters()
            )
        )

    def test_emergency_training_freezes_standard_actor(self):
        config = TrainConfig(
            **{
                **self._config().__dict__,
                "training_start_stage": "fire_extinguisher_pick_ready",
                "freeze_base_actor_for_emergency": True,
            }
        )
        trainer = PPOTrainer(config)

        self.assertTrue(
            all(
                not parameter.requires_grad
                for module in (
                    trainer.actor.encoder,
                    trainer.actor.agent_embedding,
                    trainer.actor.policy,
                )
                for parameter in module.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in trainer.actor.emergency_encoder.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in trainer.actor.emergency_policy.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for module in (
                    trainer.actor.cleanup_encoder,
                    trainer.actor.cleanup_policy,
                )
                for parameter in module.parameters()
            )
        )

    def test_fire_training_can_preserve_cleanup_actor(self):
        config = TrainConfig(
            **{
                **self._config().__dict__,
                "training_start_stage": "fire_recovery_ready",
                "freeze_base_actor_for_emergency": True,
                "freeze_cleanup_actor_for_fire": True,
            }
        )
        trainer = PPOTrainer(config)

        self.assertTrue(
            all(
                not parameter.requires_grad
                for module in (
                    trainer.actor.cleanup_encoder,
                    trainer.actor.cleanup_policy,
                )
                for parameter in module.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for module in (
                    trainer.actor.emergency_encoder,
                    trainer.actor.emergency_policy,
                )
                for parameter in module.parameters()
            )
        )

    def test_suppression_training_can_freeze_other_actor_branches(self):
        config = TrainConfig(
            **{
                **self._config().__dict__,
                "training_start_stage": "fire_extinguisher_carry_ready",
                "freeze_base_actor_for_emergency": True,
                "freeze_cleanup_actor_for_fire": True,
                "freeze_emergency_actor_for_suppression": True,
            }
        )
        trainer = PPOTrainer(config)

        self.assertTrue(
            all(
                not parameter.requires_grad
                for module in (
                    trainer.actor.encoder,
                    trainer.actor.policy,
                    trainer.actor.emergency_encoder,
                    trainer.actor.emergency_policy,
                    trainer.actor.cleanup_encoder,
                    trainer.actor.cleanup_policy,
                )
                for parameter in module.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for module in (
                    trainer.actor.suppression_encoder,
                    trainer.actor.suppression_policy,
                )
                for parameter in module.parameters()
            )
        )

    def test_emergency_training_can_preserve_suppression_actor(self):
        config = TrainConfig(
            **{
                **self._config().__dict__,
                "training_start_stage": "fire_plate_drop_ready",
                "freeze_base_actor_for_emergency": True,
                "freeze_cleanup_actor_for_fire": True,
                "freeze_suppression_actor_for_emergency": True,
            }
        )
        trainer = PPOTrainer(config)

        self.assertTrue(
            all(
                not parameter.requires_grad
                for module in (
                    trainer.actor.suppression_encoder,
                    trainer.actor.suppression_policy,
                )
                for parameter in module.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for module in (
                    trainer.actor.emergency_encoder,
                    trainer.actor.emergency_policy,
                )
                for parameter in module.parameters()
            )
        )

    def test_actor_loader_expands_append_only_observation_channels(self):
        target = PPOTrainer(self._config())
        source_state = {
            name: value.detach().clone()
            for name, value in target.actor.state_dict().items()
        }
        old_first_conv = source_state["encoder.0.weight"][:, :-1].clone()
        source_state["encoder.0.weight"] = old_first_conv

        missing, unexpected = load_actor_state_dict_compatible(
            target.actor, source_state
        )

        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        loaded = target.actor.state_dict()["encoder.0.weight"]
        torch.testing.assert_close(
            loaded[:, : old_first_conv.shape[1]], old_first_conv
        )
        torch.testing.assert_close(
            loaded[:, old_first_conv.shape[1] :],
            torch.zeros_like(loaded[:, old_first_conv.shape[1] :]),
        )

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
        self.assertIn("eval_mean_washed_plates", metrics)
        self.assertIn("eval_wash_success_rate", metrics)
        self.assertIn("eval_dirty_plate_pickup_rate", metrics)
        self.assertIn("eval_mean_raw_beef_placements", metrics)
        self.assertIn("eval_raw_beef_placement_rate", metrics)
        self.assertIn("eval_second_delivery_rate", metrics)
        self.assertIn("eval_delivery_success_rate", metrics)
        self.assertIn("eval_mean_discarded_items", metrics)
        self.assertIn("eval_mean_fire_active_steps", metrics)
        self.assertIn("eval_max_fire_active_steps", metrics)
        self.assertIn("eval_mean_longest_stagnation_steps", metrics)
        self.assertIn("eval_max_longest_stagnation_steps", metrics)
        self.assertGreaterEqual(metrics["eval_mean_deliveries"], 0.0)
        self.assertGreaterEqual(metrics["eval_action_coverage"], 0.0)
        self.assertLessEqual(metrics["eval_action_coverage"], 1.0)

    def test_configured_curriculum_stages_are_evaluated_separately(self):
        config = TrainConfig(
            **{
                **self._config().__dict__,
                "training_start_stage": "dirty_plate_carry_ready",
                "training_start_mix_stage": "plate_exhausted_ready",
                "randomize_start_positions": True,
                "random_start_max_objective_distance": 2,
                "eval_episodes": 1,
            }
        )
        trainer = PPOTrainer(config)

        metrics = evaluate_configured_stages(
            trainer.actor, config, trainer.device
        )

        self.assertIn("eval_deliveries_per_minute", metrics)
        self.assertIn(
            "train_stage_eval_deliveries_per_minute", metrics
        )
        self.assertIn(
            "mix_stage_eval_deliveries_per_minute", metrics
        )
        self.assertIn(
            "curriculum_train_stage_eval_deliveries_per_minute",
            metrics,
        )
        self.assertIn(
            "curriculum_mix_stage_eval_deliveries_per_minute",
            metrics,
        )

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
        with self.assertRaisesRegex(ValueError, "one-agent PPO"):
            TrainConfig(
                algorithm="mappo",
                num_agents=2,
                training_start_stage="serve_ready",
            )
        with self.assertRaisesRegex(ValueError, "one-agent PPO"):
            TrainConfig(
                algorithm="mappo",
                num_agents=2,
                training_start_mix_stage="serve_ready",
            )
        with self.assertRaisesRegex(
            ValueError, "only positive event reward"
        ):
            TrainConfig(
                training_start_stage="standard",
                raw_beef_placed_reward=1.0,
            )
        curriculum_config = TrainConfig(
            training_start_stage="beef_pick_ready",
            raw_beef_placed_reward=1.0,
        )
        self.assertEqual(curriculum_config.raw_beef_placed_reward, 1.0)
        mixed_fire = TrainConfig(
            training_start_stage="fire_extinguisher_pick_ready",
            training_start_mix_stage="standard",
            fire_extinguished_reward=5.0,
            allow_mixed_curriculum_event_rewards=True,
        )
        self.assertEqual(mixed_fire.fire_extinguished_reward, 5.0)
        with self.assertRaisesRegex(
            ValueError, "only permit"
        ):
            TrainConfig(
                training_start_stage="dirty_plate_carry_ready",
                training_start_mix_stage="standard",
                plate_washed_reward=4.0,
                allow_mixed_curriculum_event_rewards=True,
            )
        with self.assertRaisesRegex(
            ValueError, "rollout_temperature"
        ):
            TrainConfig(rollout_temperature=0.0)
        with self.assertRaisesRegex(
            ValueError, "fire_food_handling_penalty"
        ):
            TrainConfig(fire_food_handling_penalty=1.0)
        with self.assertRaisesRegex(
            ValueError, "fire_active_penalty"
        ):
            TrainConfig(fire_active_penalty=1.0)
        with self.assertRaisesRegex(
            ValueError, "actor_anchor_kl_coef"
        ):
            TrainConfig(actor_anchor_kl_coef=-1.0)
        with self.assertRaisesRegex(
            ValueError, "requires an actor_init"
        ):
            TrainConfig(actor_anchor_kl_coef=1.0)
        with self.assertRaisesRegex(
            ValueError, "dirty_plate_counter_handling_penalty"
        ):
            TrainConfig(dirty_plate_counter_handling_penalty=1.0)

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
        self.assertEqual(burger_config.reward.raw_beef_placed, 0.0)
        self.assertEqual(burger_config.reward.dirty_plate_pickup, 0.0)
        self.assertEqual(burger_config.reward.wash_started, 0.0)
        self.assertEqual(burger_config.reward.wash_progress, 0.0)
        self.assertEqual(burger_config.reward.plate_washed, 0.0)
        self.assertEqual(burger_config.reward.fire_extinguished, 0.0)
        self.assertEqual(burger_config.reward.fire_started, -5.0)
        self.assertEqual(burger_config.reward.fire_active, -0.25)
        self.assertEqual(burger_config.reward.fire_food_handling, -2.0)
        self.assertEqual(
            burger_config.reward.dirty_plate_counter_handling,
            -0.25,
        )
        self.assertEqual(burger_config.reward.collision, -0.05)
        self.assertEqual(burger_config.reward.time_step, 0.0)
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
                "potential_scale",
                "navigation_potential_scale",
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

    def test_vector_environment_uses_requested_direct_ppo_start_stage(self):
        env = BurgerVectorEnv(
            num_envs=2,
            num_agents=1,
            burger_config=environment_config(self._config()),
            start_stage="serve_ready",
        )

        for wrapped in env.envs:
            self.assertEqual(
                wrapped.state.players[0].held_object,
                "plated_burger",
            )
        pick_drop = Action.ACTION_TO_INDEX[Action.PICK_DROP]
        self.assertTrue(
            np.all(env.available_actions[:, 0, pick_drop] == 1.0)
        )

    def test_vector_environment_mixes_adjacent_curriculum_stages(self):
        env = BurgerVectorEnv(
            num_envs=4,
            num_agents=1,
            burger_config=environment_config(self._config()),
            start_stage="plate_pick_ready",
            start_mix_stage="rack_ready",
        )

        self.assertEqual(
            [
                wrapped._env.start_stage
                for wrapped in env.envs
            ],
            [
                "plate_pick_ready",
                "rack_ready",
                "plate_pick_ready",
                "rack_ready",
            ],
        )

    def test_short_curriculum_does_not_truncate_standard_mix_envs(self):
        env = BurgerVectorEnv(
            num_envs=2,
            num_agents=1,
            burger_config=environment_config(self._config()),
            start_stage="fire_extinguisher_pick_ready",
            start_mix_stage="standard",
            training_episode_steps=2,
        )
        stay = Action.ACTION_TO_INDEX[Action.STAY]
        actions = np.full((2, 4), stay, dtype=np.int64)

        env.step(actions)
        _, _, _, dones, infos = env.step(actions)

        self.assertTrue(dones[0])
        self.assertTrue(infos[0][0]["training_truncated"])
        self.assertFalse(dones[1])
        self.assertEqual(env.episode_steps.tolist(), [0, 2])

    def test_vector_environment_randomizes_seeded_curriculum_positions(self):
        kwargs = dict(
            num_envs=8,
            num_agents=1,
            burger_config=environment_config(self._config()),
            start_stage="wash_ready",
            randomize_start_positions=True,
            seed=91,
        )
        left = BurgerVectorEnv(**kwargs)
        right = BurgerVectorEnv(**kwargs)
        left_positions = [
            wrapped.state.players[0].position
            for wrapped in left.envs
        ]
        right_positions = [
            wrapped.state.players[0].position
            for wrapped in right.envs
        ]
        self.assertEqual(left_positions, right_positions)
        self.assertGreater(len(set(left_positions)), 1)
        self.assertTrue(
            all(wrapped.state.sink.has_dirty_plate for wrapped in left.envs)
        )

    def test_dirty_plate_distance_curriculum_starts_near_sink(self):
        max_distance = 2
        env = BurgerVectorEnv(
            num_envs=16,
            num_agents=1,
            burger_config=environment_config(self._config()),
            start_stage="dirty_plate_carry_ready",
            randomize_start_positions=True,
            random_start_max_objective_distance=max_distance,
            seed=97,
        )

        for wrapped in env.envs:
            position = wrapped.state.players[0].position
            self.assertLessEqual(
                wrapped.mdp._station_distances["W"][position],
                max_distance,
            )

    def test_plate_exhausted_distance_curriculum_starts_near_return(self):
        max_distance = 3
        env = BurgerVectorEnv(
            num_envs=16,
            num_agents=1,
            burger_config=environment_config(self._config()),
            start_stage="plate_exhausted_ready",
            randomize_start_positions=True,
            random_start_max_objective_distance=max_distance,
            seed=101,
        )

        for wrapped in env.envs:
            position = wrapped.state.players[0].position
            self.assertLessEqual(
                wrapped.mdp._station_distances["R"][position],
                max_distance,
            )

    def test_distance_curriculum_requires_randomized_positions(self):
        with self.assertRaisesRegex(
            ValueError, "requires randomize_start_positions"
        ):
            TrainConfig(random_start_max_objective_distance=2)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            TrainConfig(
                randomize_start_positions=True,
                random_start_max_objective_distance=-1,
            )

    def test_curriculum_can_focus_on_known_failure_positions(self):
        candidates = ((0, 4), (0, 5))
        env = BurgerVectorEnv(
            num_envs=16,
            num_agents=1,
            burger_config=environment_config(self._config()),
            start_stage="dirty_plate_carry_ready",
            randomize_start_positions=True,
            random_start_candidate_positions=candidates,
            seed=103,
        )

        positions = {
            wrapped.state.players[0].position for wrapped in env.envs
        }
        self.assertEqual(positions, set(candidates))

    def test_curriculum_training_can_reset_before_task_horizon(self):
        env = BurgerVectorEnv(
            num_envs=1,
            num_agents=1,
            burger_config=environment_config(self._config()),
            start_stage="dirty_plate_carry_ready",
            training_episode_steps=2,
            seed=107,
        )
        stay = Action.ACTION_TO_INDEX[Action.STAY]
        actions = np.full((1, 4), stay, dtype=np.int64)
        initial_phi = env.envs[0].mdp.potential(
            env.envs[0]._state_view
        )

        _, _, first_rewards, first_dones, _ = env.step(actions)
        self.assertFalse(first_dones[0])
        _, _, second_rewards, dones, infos = env.step(actions)

        self.assertTrue(dones[0])
        self.assertTrue(infos[0][0]["training_truncated"])
        self.assertIn(
            "training_terminal_potential_correction",
            infos[0][0],
        )
        discounted_shaping = (
            float(first_rewards[0])
            + env.envs[0].mdp.config.reward.gamma
            * float(second_rewards[0])
        )
        self.assertAlmostEqual(
            discounted_shaping, -initial_phi, places=5
        )
        self.assertAlmostEqual(
            float(second_rewards[0]),
            sum(infos[0][0]["reward_breakdown"].values()),
            places=5,
        )
        self.assertEqual(env.envs[0].state.timestep, 0)
        self.assertEqual(env.episode_steps[0], 0)

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
