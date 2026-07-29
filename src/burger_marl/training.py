"""PyTorch PPO/MAPPO trainer for the authoritative burger environment.

The one-agent mode is ordinary PPO. With two or more active agents the actor
parameters are shared, each actor receives only its local observation, and the
value function receives the separate global state exposed by
``BurgerMAPPOEnv``. This is centralized training with decentralized execution
(CTDE), rather than independent PPO with a relabeled command.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from burger_marl.actions import Action, Direction
from burger_marl.mappo_env import (
    BurgerMAPPOEnv,
    LOCAL_CHANNELS,
    MAX_AGENTS,
    TERRAIN_CHANNEL,
)
from burger_marl.env import (
    BEEF_DISPENSER,
    BUN_DISPENSER,
    EXTINGUISHER,
    FLOOR,
    GRILL,
    LETTUCE_DISPENSER,
    PLATE_RACK,
    Point,
    SERVE,
    SINK,
    BurgerConfig,
    BurgerGridworld,
    BurgerPlayerState,
    BurgerRewardConfig,
    CURRICULUM_START_STAGES,
    GrillState,
    SinkState,
)


@dataclass(frozen=True)
class TrainConfig:
    """Serializable settings shared by PPO and MAPPO."""

    algorithm: str = "ppo"
    action_contract: str = "standalone_v2_adjacent_pick_drop_process"
    action_mask_contract: str = "local_visible_adjacent_context_v2"
    world_object_contract: str = "visible_counter_or_plate_v1"
    num_agents: int = 1
    seed: int = 20260728
    total_env_steps: int = 1_000_000
    num_envs: int = 32
    rollout_length: int = 256
    update_epochs: int = 4
    num_minibatches: int = 8
    learning_rate: float = 3e-4
    gamma: float = 0.999
    gae_lambda: float = 0.98
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    actor_anchor_kl_coef: float = 0.0
    rollout_temperature: float = 1.0
    max_grad_norm: float = 10.0
    hidden_size: int = 256
    horizon: int = 429
    cook_steps: int = 24
    burn_steps: int = 16
    wash_steps: int = 10
    plate_return_steps: int = 12
    correct_delivery_reward: float = 20.0
    raw_beef_placed_reward: float = 0.0
    dirty_plate_pickup_reward: float = 0.0
    wash_started_reward: float = 0.0
    wash_progress_reward: float = 0.0
    plate_washed_reward: float = 0.0
    fire_extinguished_reward: float = 0.0
    fire_started_penalty: float = -5.0
    fire_active_penalty: float = -0.25
    fire_food_handling_penalty: float = -2.0
    dirty_plate_counter_handling_penalty: float = -0.25
    collision_penalty: float = -0.05
    time_step_penalty: float = 0.0
    potential_scale: float = 1.0
    navigation_potential_scale: float = 1.0
    max_all_agents_stay_ratio: float = 0.98
    max_all_agents_noop_ratio: float = 0.995
    action_freeze_patience_updates: int = 3
    eval_episodes: int = 5
    eval_interval_updates: int = 20
    checkpoint_interval_updates: int = 20
    device: str = "auto"
    output_dir: str = "runs/burger"
    actor_init: Optional[str] = None
    deterministic_torch: bool = False
    training_start_stage: str = "standard"
    training_start_mix_stage: Optional[str] = None
    randomize_start_positions: bool = False
    random_start_max_objective_distance: Optional[int] = None
    random_start_candidate_positions: Tuple[Tuple[int, int], ...] = ()
    training_episode_steps: Optional[int] = None
    allow_mixed_curriculum_event_rewards: bool = False
    freeze_base_actor_for_emergency: bool = False
    freeze_cleanup_actor_for_fire: bool = False
    freeze_emergency_actor_for_suppression: bool = False
    freeze_suppression_actor_for_emergency: bool = False
    freeze_base_actor_for_dish: bool = False
    freeze_fire_actors_for_dish: bool = False
    freeze_dish_actor_for_assembly: bool = False
    freeze_assembly_actor_for_workflow: bool = False
    bc_pretrain_steps: int = 0
    bc_batch_size: int = 128
    bc_learning_rate: float = 1e-3
    bc_aux_coef: float = 0.0

    def __post_init__(self) -> None:
        if self.algorithm not in {"ppo", "mappo"}:
            raise ValueError("algorithm must be ppo or mappo")
        if self.action_contract != "standalone_v2_adjacent_pick_drop_process":
            raise ValueError("Unsupported burger action contract")
        if self.action_mask_contract != "local_visible_adjacent_context_v2":
            raise ValueError("Unsupported burger action-mask contract")
        if self.world_object_contract != "visible_counter_or_plate_v1":
            raise ValueError("Unsupported burger world-object contract")
        if self.training_start_stage not in CURRICULUM_START_STAGES:
            raise ValueError("Unsupported training_start_stage")
        if (
            self.training_start_mix_stage is not None
            and self.training_start_mix_stage
            not in CURRICULUM_START_STAGES
        ):
            raise ValueError("Unsupported training_start_mix_stage")
        if (
            (
                self.training_start_stage != "standard"
                or self.training_start_mix_stage is not None
            )
            and self.num_agents != 1
        ):
            raise ValueError(
                "Curriculum start states support one-agent PPO only"
            )
        if (
            self.random_start_max_objective_distance is not None
            and self.random_start_max_objective_distance < 0
        ):
            raise ValueError(
                "random_start_max_objective_distance must be non-negative"
            )
        if (
            self.random_start_max_objective_distance is not None
            and not self.randomize_start_positions
        ):
            raise ValueError(
                "random_start_max_objective_distance requires "
                "randomize_start_positions"
            )
        normalized_candidates = tuple(
            tuple(position)
            for position in self.random_start_candidate_positions
        )
        object.__setattr__(
            self,
            "random_start_candidate_positions",
            normalized_candidates,
        )
        if any(
            len(position) != 2
            or any(not isinstance(value, int) for value in position)
            for position in normalized_candidates
        ):
            raise ValueError(
                "random_start_candidate_positions must contain integer "
                "(x, y) pairs"
            )
        if (
            normalized_candidates
            and not self.randomize_start_positions
        ):
            raise ValueError(
                "random_start_candidate_positions requires "
                "randomize_start_positions"
            )
        if (
            self.training_episode_steps is not None
            and self.training_episode_steps <= 0
        ):
            raise ValueError("training_episode_steps must be positive")
        if (
            self.training_episode_steps is not None
            and self.training_start_stage == "standard"
        ):
            raise ValueError(
                "training_episode_steps is reserved for curriculum starts"
            )
        if self.algorithm == "ppo" and self.num_agents != 1:
            raise ValueError("PPO mode requires exactly one active agent")
        if self.algorithm == "mappo" and self.num_agents < 2:
            raise ValueError("MAPPO mode requires at least two active agents")
        if not 1 <= self.num_agents <= MAX_AGENTS:
            raise ValueError("num_agents must be between one and four")
        positive = {
            "total_env_steps": self.total_env_steps,
            "num_envs": self.num_envs,
            "rollout_length": self.rollout_length,
            "update_epochs": self.update_epochs,
            "num_minibatches": self.num_minibatches,
            "learning_rate": self.learning_rate,
            "rollout_temperature": self.rollout_temperature,
            "max_grad_norm": self.max_grad_norm,
            "hidden_size": self.hidden_size,
            "horizon": self.horizon,
            "cook_steps": self.cook_steps,
            "burn_steps": self.burn_steps,
            "wash_steps": self.wash_steps,
            "plate_return_steps": self.plate_return_steps,
            "eval_episodes": self.eval_episodes,
            "eval_interval_updates": self.eval_interval_updates,
            "checkpoint_interval_updates": self.checkpoint_interval_updates,
            "bc_batch_size": self.bc_batch_size,
            "bc_learning_rate": self.bc_learning_rate,
            "action_freeze_patience_updates": self.action_freeze_patience_updates,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError("Training values must be positive: {}".format(invalid))
        if self.fire_started_penalty > 0:
            raise ValueError("fire_started_penalty must not be positive")
        if self.fire_active_penalty > 0:
            raise ValueError("fire_active_penalty must not be positive")
        if self.fire_food_handling_penalty > 0:
            raise ValueError(
                "fire_food_handling_penalty must not be positive"
            )
        if self.dirty_plate_counter_handling_penalty > 0:
            raise ValueError(
                "dirty_plate_counter_handling_penalty must not be positive"
            )
        if self.collision_penalty > 0:
            raise ValueError("collision_penalty must not be positive")
        if self.time_step_penalty > 0:
            raise ValueError("time_step_penalty must not be positive")
        if self.correct_delivery_reward <= 0:
            raise ValueError("correct_delivery_reward must be positive")
        non_negative_task_rewards = {
            "raw_beef_placed_reward": self.raw_beef_placed_reward,
            "dirty_plate_pickup_reward": self.dirty_plate_pickup_reward,
            "wash_started_reward": self.wash_started_reward,
            "wash_progress_reward": self.wash_progress_reward,
            "plate_washed_reward": self.plate_washed_reward,
            "fire_extinguished_reward": self.fire_extinguished_reward,
        }
        if any(value < 0 for value in non_negative_task_rewards.values()):
            raise ValueError("Task milestone rewards must not be negative")
        stages = {
            self.training_start_stage,
            self.training_start_mix_stage,
        }
        positive_milestones = {
            name
            for name, value in non_negative_task_rewards.items()
            if value > 0
        }
        mixed_fire_override_is_valid = (
            self.allow_mixed_curriculum_event_rewards
            and "standard" in stages
            and any(
                stage is not None and stage.startswith("fire_")
                for stage in stages
            )
            and positive_milestones == {"fire_extinguished_reward"}
        )
        if (
            self.allow_mixed_curriculum_event_rewards
            and not mixed_fire_override_is_valid
        ):
            raise ValueError(
                "Mixed curriculum event rewards only permit a temporary "
                "fire_extinguished reward in a standard-plus-fire curriculum"
            )
        if (
            self.freeze_base_actor_for_emergency
            and not any(
                stage is not None and stage.startswith("fire_")
                for stage in stages
            )
        ):
            raise ValueError(
                "freeze_base_actor_for_emergency requires a fire curriculum"
            )
        if (
            self.freeze_cleanup_actor_for_fire
            and not any(
                stage is not None and stage.startswith("fire_")
                for stage in stages
            )
        ):
            raise ValueError(
                "freeze_cleanup_actor_for_fire requires a fire curriculum"
            )
        if (
            self.freeze_emergency_actor_for_suppression
            and not any(
                stage is not None and stage.startswith("fire_")
                for stage in stages
            )
        ):
            raise ValueError(
                "freeze_emergency_actor_for_suppression requires a fire "
                "curriculum"
            )
        if (
            self.freeze_suppression_actor_for_emergency
            and not any(
                stage is not None and stage.startswith("fire_")
                for stage in stages
            )
        ):
            raise ValueError(
                "freeze_suppression_actor_for_emergency requires a fire "
                "curriculum"
            )
        dish_curriculum = any(
            stage in {
                "dirty_plate_carry_ready",
                "plate_exhausted_ready",
            }
            for stage in stages
        )
        if self.freeze_base_actor_for_dish and not dish_curriculum:
            raise ValueError(
                "freeze_base_actor_for_dish requires a dish curriculum"
            )
        if self.freeze_fire_actors_for_dish and not dish_curriculum:
            raise ValueError(
                "freeze_fire_actors_for_dish requires a dish curriculum"
            )
        if self.freeze_dish_actor_for_assembly and not dish_curriculum:
            raise ValueError(
                "freeze_dish_actor_for_assembly requires a dish curriculum"
            )
        if self.freeze_assembly_actor_for_workflow and not dish_curriculum:
            raise ValueError(
                "freeze_assembly_actor_for_workflow requires a dish curriculum"
            )
        if (
            "standard" in stages
            and positive_milestones
            and not mixed_fire_override_is_valid
        ):
            raise ValueError(
                "Standard-start training must use delivery as its only "
                "positive event reward unless the explicit mixed fire "
                "curriculum override is enabled"
            )
        if self.potential_scale < 0:
            raise ValueError("potential_scale must not be negative")
        if self.navigation_potential_scale < 0:
            raise ValueError(
                "navigation_potential_scale must not be negative"
            )
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if not 0 <= self.gae_lambda <= 1:
            raise ValueError("gae_lambda must be in [0, 1]")
        if not 0 < self.clip_coef < 1:
            raise ValueError("clip_coef must be in (0, 1)")
        if self.value_coef < 0:
            raise ValueError("value_coef must not be negative")
        if self.entropy_coef < 0:
            raise ValueError("entropy_coef must not be negative")
        if self.actor_anchor_kl_coef < 0:
            raise ValueError("actor_anchor_kl_coef must not be negative")
        if self.actor_anchor_kl_coef > 0 and not self.actor_init:
            raise ValueError(
                "actor_anchor_kl_coef requires an actor_init checkpoint"
            )
        if not 0 <= self.max_all_agents_stay_ratio <= 1:
            raise ValueError("max_all_agents_stay_ratio must be in [0, 1]")
        if not 0 <= self.max_all_agents_noop_ratio <= 1:
            raise ValueError("max_all_agents_noop_ratio must be in [0, 1]")
        if self.bc_pretrain_steps < 0:
            raise ValueError("bc_pretrain_steps must not be negative")
        if self.bc_aux_coef < 0:
            raise ValueError("bc_aux_coef must not be negative")
        if self.bc_aux_coef > 0 and self.bc_pretrain_steps == 0:
            raise ValueError("bc_aux_coef requires behavior-cloning pretraining")
        if self.bc_pretrain_steps > 0 and self.num_agents != 1:
            raise ValueError(
                "Behavior-cloning pretraining supports one-agent PPO only"
            )
        if self.num_minibatches > self.num_envs * self.rollout_length:
            raise ValueError("num_minibatches exceeds rollout transitions")


class LocalActor(nn.Module):
    """Local actor with an isolated, locally gated emergency residual."""

    def __init__(
        self,
        observation_shape: Sequence[int],
        hidden_size: int,
        num_actions: int,
    ) -> None:
        super().__init__()
        channels, height, width = observation_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        encoded_size = 64 * height * width
        self.agent_embedding = nn.Embedding(MAX_AGENTS, 8)
        self.policy = nn.Sequential(
            nn.Linear(encoded_size + 8, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_actions),
        )
        self.emergency_encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.emergency_policy = nn.Sequential(
            nn.Linear(encoded_size + 8, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_actions),
        )
        self.cleanup_encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.cleanup_policy = nn.Sequential(
            nn.Linear(encoded_size + 8, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_actions),
        )
        self.suppression_encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.suppression_policy = nn.Sequential(
            nn.Linear(encoded_size + 8, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_actions),
        )
        self.dish_encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.dish_policy = nn.Sequential(
            nn.Linear(encoded_size + 8, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_actions),
        )
        self.assembly_encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.assembly_policy = nn.Sequential(
            nn.Linear(encoded_size + 8, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_actions),
        )
        self.workflow_encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.workflow_policy = nn.Sequential(
            nn.Linear(encoded_size + 8, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_actions),
        )
        self.apply(_orthogonal_init)
        nn.init.orthogonal_(self.policy[-1].weight, gain=0.01)
        nn.init.zeros_(self.emergency_policy[-1].weight)
        nn.init.zeros_(self.emergency_policy[-1].bias)
        nn.init.zeros_(self.cleanup_policy[-1].weight)
        nn.init.zeros_(self.cleanup_policy[-1].bias)
        nn.init.zeros_(self.suppression_policy[-1].weight)
        nn.init.zeros_(self.suppression_policy[-1].bias)
        nn.init.zeros_(self.dish_policy[-1].weight)
        nn.init.zeros_(self.dish_policy[-1].bias)
        nn.init.zeros_(self.assembly_policy[-1].weight)
        nn.init.zeros_(self.assembly_policy[-1].bias)
        nn.init.zeros_(self.workflow_policy[-1].weight)
        nn.init.zeros_(self.workflow_policy[-1].bias)

    def forward(
        self, observations: torch.Tensor, agent_ids: torch.Tensor
    ) -> torch.Tensor:
        features = self.encoder(observations)
        identity = self.agent_embedding(agent_ids)
        base_logits = self.policy(torch.cat((features, identity), dim=-1))
        emergency_features = self.emergency_encoder(observations)
        emergency_logits = self.emergency_policy(
            torch.cat((emergency_features, identity), dim=-1)
        )
        cleanup_features = self.cleanup_encoder(observations)
        cleanup_logits = self.cleanup_policy(
            torch.cat((cleanup_features, identity), dim=-1)
        )
        suppression_features = self.suppression_encoder(observations)
        suppression_logits = self.suppression_policy(
            torch.cat((suppression_features, identity), dim=-1)
        )
        dish_features = self.dish_encoder(observations)
        dish_logits = self.dish_policy(
            torch.cat((dish_features, identity), dim=-1)
        )
        assembly_features = self.assembly_encoder(observations)
        assembly_logits = self.assembly_policy(
            torch.cat((assembly_features, identity), dim=-1)
        )
        workflow_features = self.workflow_encoder(observations)
        workflow_logits = self.workflow_policy(
            torch.cat((workflow_features, identity), dim=-1)
        )
        fire_alarm = observations[
            :, LOCAL_CHANNELS["fire_alarm"]
        ].amax(dim=(1, 2))
        extinguisher = observations[:, LOCAL_CHANNELS["extinguisher"]]
        extinguisher_station = observations[
            :, TERRAIN_CHANNEL[EXTINGUISHER]
        ]
        held_or_misplaced_extinguisher = (
            extinguisher * (1.0 - extinguisher_station)
        ).amax(dim=(1, 2))
        suppression_gate = (
            fire_alarm * held_or_misplaced_extinguisher
        ).clamp(0.0, 1.0)
        fire_gate = (
            fire_alarm * (1.0 - held_or_misplaced_extinguisher)
        ).clamp(0.0, 1.0)
        cleanup_gate = (
            (1.0 - fire_alarm) * held_or_misplaced_extinguisher
        ).clamp(0.0, 1.0)
        plate_shortage = observations[
            :, LOCAL_CHANNELS["plate_shortage"]
        ].amax(dim=(1, 2))
        dirty_plate_map = observations[
            :, LOCAL_CHANNELS["dirty_plate"]
        ]
        center_y = observations.shape[-2] // 2
        center_x = observations.shape[-1] // 2
        held_dirty_plate = dirty_plate_map[:, center_y, center_x]
        dirty_plate_in_sink = (
            dirty_plate_map * observations[:, TERRAIN_CHANNEL[SINK]]
        ).amax(dim=(1, 2))
        active_dirty_plate = torch.maximum(
            held_dirty_plate, dirty_plate_in_sink
        )
        dish_gate = (
            plate_shortage
            * active_dirty_plate
            * (1.0 - fire_alarm)
            * (1.0 - held_or_misplaced_extinguisher)
        ).clamp(0.0, 1.0)
        usable_plate_visible = observations[
            :, LOCAL_CHANNELS["plate"]
        ].amax(dim=(1, 2))
        assembly_gate = (
            plate_shortage
            * (1.0 - active_dirty_plate)
            * usable_plate_visible
            * (1.0 - fire_alarm)
            * (1.0 - held_or_misplaced_extinguisher)
        ).clamp(0.0, 1.0)
        held_plate = observations[
            :, LOCAL_CHANNELS["plate"], center_y, center_x
        ]
        held_bun = observations[
            :, LOCAL_CHANNELS["bun"], center_y, center_x
        ]
        held_lettuce = observations[
            :, LOCAL_CHANNELS["lettuce"], center_y, center_x
        ]
        held_cooked_beef = observations[
            :, LOCAL_CHANNELS["cooked_beef"], center_y, center_x
        ]
        held_bun_lettuce_plate = (
            held_plate
            * held_bun
            * held_lettuce
            * (1.0 - held_cooked_beef)
        )
        workflow_gate = (
            plate_shortage
            * held_bun_lettuce_plate
            * (1.0 - fire_alarm)
            * (1.0 - held_or_misplaced_extinguisher)
        ).clamp(0.0, 1.0)
        return (
            base_logits
            + fire_gate.unsqueeze(-1) * emergency_logits
            + suppression_gate.unsqueeze(-1) * suppression_logits
            + cleanup_gate.unsqueeze(-1) * cleanup_logits
            + dish_gate.unsqueeze(-1) * dish_logits
            + assembly_gate.unsqueeze(-1) * assembly_logits
            + workflow_gate.unsqueeze(-1) * workflow_logits
        )


def load_actor_state_dict_compatible(
    actor: LocalActor,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    """Load actors across append-only local-observation channel upgrades."""

    adapted = dict(state_dict)
    for first_conv in (
        "encoder.0.weight",
        "emergency_encoder.0.weight",
        "cleanup_encoder.0.weight",
        "suppression_encoder.0.weight",
        "dish_encoder.0.weight",
        "assembly_encoder.0.weight",
        "workflow_encoder.0.weight",
    ):
        source_weight = adapted.get(first_conv)
        target_weight = actor.state_dict()[first_conv]
        if (
            source_weight is not None
            and source_weight.shape != target_weight.shape
            and source_weight.ndim == target_weight.ndim == 4
            and source_weight.shape[0] == target_weight.shape[0]
            and source_weight.shape[2:] == target_weight.shape[2:]
            and source_weight.shape[1] < target_weight.shape[1]
        ):
            expanded = torch.zeros_like(target_weight)
            expanded[:, : source_weight.shape[1]] = source_weight.to(
                device=expanded.device,
                dtype=expanded.dtype,
            )
            adapted[first_conv] = expanded
    for target_name in actor.state_dict():
        if (
            target_name.startswith("suppression_")
            and target_name not in adapted
        ):
            source_name = target_name.replace(
                "suppression_", "emergency_", 1
            )
            if source_name in adapted:
                adapted[target_name] = adapted[source_name].clone()
    incompatible = actor.load_state_dict(adapted, strict=strict)
    return list(incompatible.missing_keys), list(
        incompatible.unexpected_keys
    )


class CentralCritic(nn.Module):
    """One team value from the full state; never exported with the actor."""

    def __init__(self, shared_observation_size: int, hidden_size: int) -> None:
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(shared_observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.apply(_orthogonal_init)
        nn.init.orthogonal_(self.value[-1].weight, gain=1.0)

    def forward(self, shared_observations: torch.Tensor) -> torch.Tensor:
        return self.value(shared_observations).squeeze(-1)


def _orthogonal_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _masked_logits(
    logits: torch.Tensor, available_actions: torch.Tensor
) -> torch.Tensor:
    """Exclude invalid world-aligned movement and adjacent work actions."""

    if logits.shape != available_actions.shape:
        raise ValueError("Action-mask shape must match actor logits")
    if not torch.all(torch.isfinite(available_actions)):
        raise ValueError("Action mask must contain only finite values")
    if torch.any(
        (available_actions != 0) & (available_actions != 1)
    ):
        raise ValueError("Action mask must be binary")
    if torch.any(available_actions.sum(dim=-1) < 1):
        raise ValueError("Every active actor needs at least one legal action")
    return logits.masked_fill(
        available_actions <= 0,
        torch.finfo(logits.dtype).min,
    )


@dataclass
class Rollout:
    observations: torch.Tensor
    shared_observations: torch.Tensor
    available_actions: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    completed_returns: List[float]
    completed_deliveries: List[int]
    collision_events: int
    fire_events: int
    action_counts: Tuple[int, ...]
    all_agents_stay_steps: int
    all_agents_noop_steps: int
    joint_steps: int
    legal_action_slots: int
    actor_slots: int
    reward_component_totals: Dict[str, float]
    environment_seconds: float

    @property
    def all_agents_stay_ratio(self) -> float:
        return self.all_agents_stay_steps / max(self.joint_steps, 1)

    @property
    def all_agents_noop_ratio(self) -> float:
        return self.all_agents_noop_steps / max(self.joint_steps, 1)

    @property
    def mean_legal_actions(self) -> float:
        return self.legal_action_slots / max(self.actor_slots, 1)


@dataclass(frozen=True)
class ExpertDemo:
    """One legal fixed-start trajectory through the complete burger loop."""

    observations: np.ndarray
    available_actions: np.ndarray
    actions: np.ndarray
    delivered_orders: int


def environment_config(config: TrainConfig) -> BurgerConfig:
    """Materialize every task parameter recorded in a training checkpoint."""

    return BurgerConfig(
        horizon=config.horizon,
        cook_steps=config.cook_steps,
        burn_steps=config.burn_steps,
        wash_steps=config.wash_steps,
        plate_return_steps=config.plate_return_steps,
        reward=BurgerRewardConfig(
            correct_delivery=config.correct_delivery_reward,
            raw_beef_placed=config.raw_beef_placed_reward,
            dirty_plate_pickup=config.dirty_plate_pickup_reward,
            wash_started=config.wash_started_reward,
            wash_progress=config.wash_progress_reward,
            plate_washed=config.plate_washed_reward,
            fire_extinguished=config.fire_extinguished_reward,
            fire_started=config.fire_started_penalty,
            fire_active=config.fire_active_penalty,
            fire_food_handling=config.fire_food_handling_penalty,
            dirty_plate_counter_handling=(
                config.dirty_plate_counter_handling_penalty
            ),
            collision=config.collision_penalty,
            time_step=config.time_step_penalty,
            potential_scale=config.potential_scale,
            navigation_potential_scale=(
                config.navigation_potential_scale
            ),
            gamma=config.gamma,
        ),
    )


def audit_training_contract(config: TrainConfig) -> Dict[str, object]:
    """Fail closed unless every PPO action and reward source is coherent."""

    burger_config = environment_config(config)
    mdp = BurgerGridworld(config=burger_config)
    expected_actions = (
        Direction.NORTH,
        Direction.SOUTH,
        Direction.EAST,
        Direction.WEST,
        Action.STAY,
        Action.PICK_DROP,
        Action.PROCESS,
    )
    if tuple(Action.INDEX_TO_ACTION) != expected_actions:
        raise RuntimeError("Seven-action index contract changed")

    reward_keys = {
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
    }

    def assert_reward(transition: object) -> Dict[str, float]:
        breakdown = dict(transition.info["reward_breakdown"])
        if set(breakdown) != reward_keys:
            raise RuntimeError("Reward breakdown contract changed")
        if not all(math.isfinite(float(value)) for value in breakdown.values()):
            raise RuntimeError("Reward breakdown contains a non-finite value")
        if not math.isclose(
            float(transition.reward),
            sum(float(value) for value in breakdown.values()),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError("Environment reward does not equal its breakdown")
        return breakdown

    def station_position(terrain: str) -> Tuple[int, int]:
        matches = [
            (x, y)
            for y, row in enumerate(mdp.layout.rows)
            for x, cell in enumerate(row)
            if cell == terrain
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "Expected one {!r} station, got {}".format(terrain, matches)
            )
        return matches[0]

    def adjacent_floor(target: Tuple[int, int]) -> Tuple[int, int]:
        matches = [
            Action.move_in_direction(target, direction)
            for direction in Direction.ALL_DIRECTIONS
            if mdp.layout.terrain_at(
                Action.move_in_direction(target, direction)
            )
            == FLOOR
        ]
        if not matches:
            raise RuntimeError("No adjacent floor for station {}".format(target))
        return matches[0]

    action_effects: Dict[str, str] = {}
    for name, direction in (
        ("north", Direction.NORTH),
        ("south", Direction.SOUTH),
        ("east", Direction.EAST),
        ("west", Direction.WEST),
    ):
        move_pair = next(
            (
                (source, Action.move_in_direction(source, direction))
                for source in sorted(mdp.layout.valid_player_positions)
                if Action.move_in_direction(source, direction)
                in mdp.layout.valid_player_positions
            ),
            None,
        )
        if move_pair is None:
            raise RuntimeError("No representative {} movement".format(name))
        state = mdp.get_standard_start_state(1)
        state.players[0].position = move_pair[0]
        transition = mdp.get_state_transition(state, [direction])
        assert_reward(transition)
        if transition.state.players[0].position != move_pair[1]:
            raise RuntimeError("{} movement did not advance one cell".format(name))
        action_effects[name] = "one_cardinal_floor_cell"

    state = mdp.get_standard_start_state(1)
    old_position = state.players[0].position
    stayed = mdp.get_state_transition(state, [Action.STAY])
    stay_reward = assert_reward(stayed)
    if stayed.state.players[0].position != old_position:
        raise RuntimeError("STAY changed player position")
    if stay_reward["time_step"] != burger_config.reward.time_step:
        raise RuntimeError("STAY did not receive the configured time cost")
    action_effects["stay"] = "position_unchanged_environment_clock_advances"

    picked = mdp.get_state_transition(
        mdp.get_standard_start_state(1),
        [Action.PICK_DROP],
    )
    pickup_reward = assert_reward(picked)
    if picked.state.players[0].held_object != "bun":
        raise RuntimeError("PICK_DROP did not pick the adjacent bun")
    if "dispenser_pickup" not in {
        str(event["type"]) for event in picked.info["events"]
    }:
        raise RuntimeError("PICK_DROP did not emit its physical transfer event")
    if pickup_reward["correct_delivery"] != 0:
        raise RuntimeError("Ingredient pickup received sparse delivery reward")
    action_effects["pick_drop"] = "one_adjacent_physical_transfer"

    repeated = mdp.get_state_transition(
        picked.state,
        [Action.PICK_DROP],
    )
    repeated_reward = assert_reward(repeated)
    if repeated.info["events"]:
        raise RuntimeError("Invalid repeated interaction emitted an event")
    if any(
        repeated_reward[name] != 0
        for name in ("correct_delivery", "fire_started", "collision")
    ):
        raise RuntimeError("Invalid repeated interaction received event reward")
    if repeated.reward > 0:
        raise RuntimeError("Invalid repeated interaction produced positive reward")

    sink_state = mdp.get_standard_start_state(1)
    sink_state.players[0].position = adjacent_floor(
        station_position(SINK)
    )
    sink_state.clean_plates -= 1
    sink_state.sink = SinkState(
        has_dirty_plate=True,
        wash_progress=0,
    )
    washed = mdp.get_state_transition(sink_state, [Action.PROCESS])
    assert_reward(washed)
    if washed.state.sink.wash_progress != 1:
        raise RuntimeError("PROCESS did not advance washing by exactly one tick")

    wash_interrupted = washed.state
    wash_interrupted.players[0].position = adjacent_floor(
        station_position(BUN_DISPENSER)
    )
    fetched_during_interrupted_wash = mdp.get_state_transition(
        wash_interrupted, [Action.PICK_DROP]
    )
    assert_reward(fetched_during_interrupted_wash)
    if (
        fetched_during_interrupted_wash.state.players[0].held_object
        != "bun"
        or fetched_during_interrupted_wash.state.sink.wash_progress != 1
    ):
        raise RuntimeError(
            "Interrupted wash incorrectly reserved or froze an agent"
        )

    fire_state = mdp.get_standard_start_state(1)
    fire_state.players[0] = BurgerPlayerState(
        adjacent_floor(station_position(GRILL)),
        Direction.NORTH,
        "extinguisher",
    )
    fire_state.extinguisher_available = False
    fire_state.grill = GrillState(
        food="burnt_beef",
        cook_ticks=burger_config.cook_steps,
        ready_ticks=burger_config.burn_steps,
    )
    extinguished = mdp.get_state_transition(
        fire_state, [Action.PROCESS]
    )
    assert_reward(extinguished)
    if extinguished.state.grill.food is not None:
        raise RuntimeError("PROCESS did not clear the burning grill")
    action_effects["process"] = "one_adjacent_wash_tick_or_fire_suppression"

    delivery_state = mdp.get_standard_start_state(1)
    delivery_state.players[0] = BurgerPlayerState(
        adjacent_floor(station_position(SERVE)),
        Direction.NORTH,
        "plated_burger",
    )
    delivery_state.clean_plates -= 1
    delivered = mdp.get_state_transition(
        delivery_state, [Action.PICK_DROP]
    )
    delivery_reward = assert_reward(delivered)
    if (
        delivered.state.delivered_orders != 1
        or delivery_reward["correct_delivery"]
        != burger_config.reward.correct_delivery
    ):
        raise RuntimeError("Correct delivery sparse reward is inconsistent")

    fire_started_state = mdp.get_standard_start_state(1)
    fire_started_state.grill = GrillState(
        food="cooked_beef",
        cook_ticks=burger_config.cook_steps,
        ready_ticks=burger_config.burn_steps - 1,
    )
    fire_started = mdp.get_state_transition(
        fire_started_state, [Action.STAY]
    )
    fire_reward = assert_reward(fire_started)
    if fire_reward["fire_started"] != burger_config.reward.fire_started:
        raise RuntimeError("Fire penalty is not applied exactly once")

    collision_state = mdp.get_standard_start_state(3)
    collision_state.players[0].position = (0, 1)
    collision_state.players[1].position = (2, 1)
    collision_state.players[2].position = (0, 2)
    collision = mdp.get_state_transition(
        collision_state,
        [Direction.EAST, Direction.WEST, Direction.EAST],
    )
    collision_reward = assert_reward(collision)
    if collision_reward["collision"] != burger_config.reward.collision:
        raise RuntimeError("Collision penalty does not match configuration")
    if collision.state.players[2].position != (1, 2):
        raise RuntimeError("A local collision froze an unrelated agent")

    initial = mdp.get_standard_start_state(1)
    old_phi = mdp.potential(initial)
    shaped = mdp.get_state_transition(initial, [Action.PICK_DROP])
    shaped_reward = assert_reward(shaped)
    expected_shaping = burger_config.reward.potential_scale * (
        config.gamma * mdp.potential(shaped.state) - old_phi
    )
    if not math.isclose(
        shaped_reward["potential"],
        expected_shaping,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("Potential shaping gamma differs from PPO gamma")

    for active_agents in range(1, MAX_AGENTS + 1):
        adapter = BurgerMAPPOEnv(
            mdp=BurgerGridworld(config=burger_config),
            num_players=active_agents,
        )
        _, _, masks = adapter.reset(seed=config.seed)
        stay_index = Action.ACTION_TO_INDEX[Action.STAY]
        active_masks = masks[:active_agents]
        if np.any(active_masks.sum(axis=-1) < 1):
            raise RuntimeError("An active agent received an all-zero action mask")
        if np.any(active_masks[:, stay_index] != 1):
            raise RuntimeError("STAY must remain legal for every active agent")
        inactive_masks = masks[active_agents:]
        if inactive_masks.size and (
            np.any(inactive_masks[:, stay_index] != 1)
            or np.any(inactive_masks.sum(axis=-1) != 1)
        ):
            raise RuntimeError("Inactive agent masks must contain only STAY")

    vector_probe = BurgerVectorEnv(
        num_envs=1,
        num_agents=1,
        burger_config=burger_config,
    )
    pick_drop_index = Action.ACTION_TO_INDEX[Action.PICK_DROP]
    stay_index = Action.ACTION_TO_INDEX[Action.STAY]
    if vector_probe.available_actions[0, 0, pick_drop_index] != 1:
        raise RuntimeError("Initial adjacent pickup is unexpectedly masked")
    probe_actions = np.full((1, MAX_AGENTS), stay_index, dtype=np.int64)
    probe_actions[0, 0] = pick_drop_index
    vector_probe.step(probe_actions)
    if vector_probe.available_actions[0, 0, pick_drop_index] != 0:
        raise RuntimeError("Vector environment retained a stale action mask")

    return {
        "status": "passed",
        "action_count": len(action_effects),
        "actions": action_effects,
        "reward_components": {
            "correct_delivery": burger_config.reward.correct_delivery,
            "raw_beef_placed": burger_config.reward.raw_beef_placed,
            "dirty_plate_pickup": burger_config.reward.dirty_plate_pickup,
            "wash_started": burger_config.reward.wash_started,
            "wash_progress": burger_config.reward.wash_progress,
            "plate_washed": burger_config.reward.plate_washed,
            "fire_extinguished": burger_config.reward.fire_extinguished,
            "fire_started": burger_config.reward.fire_started,
            "fire_active": burger_config.reward.fire_active,
            "fire_food_handling": (
                burger_config.reward.fire_food_handling
            ),
            "dirty_plate_counter_handling": (
                burger_config.reward.dirty_plate_counter_handling
            ),
            "collision": burger_config.reward.collision,
            "time_step": burger_config.reward.time_step,
            "potential_scale": burger_config.reward.potential_scale,
            "navigation_potential_scale": (
                burger_config.reward.navigation_potential_scale
            ),
            "potential_gamma": burger_config.reward.gamma,
        },
        "active_masks_never_empty": True,
        "vector_masks_refresh_after_step": True,
        "unrelated_agents_continue_after_local_collision": True,
        "invalid_interactions_have_no_positive_reward": True,
        "potential_gamma_matches_ppo_gamma": True,
        "standard_objective_positive_event_is_delivery_only": (
            "standard"
            not in {
                config.training_start_stage,
                config.training_start_mix_stage,
            }
            or all(
                value == 0
                for value in (
                    burger_config.reward.raw_beef_placed,
                    burger_config.reward.dirty_plate_pickup,
                    burger_config.reward.wash_started,
                    burger_config.reward.wash_progress,
                    burger_config.reward.plate_washed,
                    burger_config.reward.fire_extinguished,
                )
            )
        ),
        "mixed_curriculum_event_reward_override": (
            config.allow_mixed_curriculum_event_rewards
        ),
        "base_actor_frozen_for_emergency": (
            config.freeze_base_actor_for_emergency
        ),
        "cleanup_actor_frozen_for_fire": (
            config.freeze_cleanup_actor_for_fire
        ),
        "emergency_actor_frozen_for_suppression": (
            config.freeze_emergency_actor_for_suppression
        ),
        "suppression_actor_frozen_for_emergency": (
            config.freeze_suppression_actor_for_emergency
        ),
        "base_actor_frozen_for_dish": config.freeze_base_actor_for_dish,
        "fire_actors_frozen_for_dish": (
            config.freeze_fire_actors_for_dish
        ),
        "wash_progress_does_not_reserve_agent": True,
        "sink_process_mask_has_no_hidden_owner": True,
        "ppo_rollout_temperature": config.rollout_temperature,
        "actor_anchor_kl_coef": config.actor_anchor_kl_coef,
    }


def generate_single_agent_expert_demo(config: TrainConfig) -> ExpertDemo:
    """Generate a shortest-path legal demonstration in the real environment.

    The demonstrator has no privileged transition or reward API. It submits
    the same seven actions as PPO, records the same local observations and
    action masks, and must pass the normal fixed-start delivery transition.
    """

    if config.num_agents != 1:
        raise ValueError("The stage-one expert supports one active agent")
    env = BurgerMAPPOEnv(
        mdp=BurgerGridworld(config=environment_config(config)),
        num_players=1,
    )
    observations, _, available = env.reset(seed=config.seed)
    demo_observations: List[np.ndarray] = []
    demo_available: List[np.ndarray] = []
    demo_actions: List[int] = []
    stay_index = Action.ACTION_TO_INDEX[Action.STAY]

    def submit(action: object) -> None:
        nonlocal observations, available
        action_index = Action.ACTION_TO_INDEX[action]
        if available[0, action_index] != 1.0:
            raise RuntimeError(
                "Expert attempted masked action {!r}".format(action)
            )
        demo_observations.append(observations[0].copy())
        demo_available.append(available[0].copy())
        demo_actions.append(action_index)
        joint = np.full(MAX_AGENTS, stay_index, dtype=np.int64)
        joint[0] = action_index
        (
            observations,
            _,
            _,
            dones,
            _,
            available,
        ) = env.step(joint)
        if dones[0] and env.state.delivered_orders == 0:
            raise RuntimeError("Expert trajectory exhausted the episode")

    def terrain_position(terrain: str) -> Tuple[int, int]:
        matches = [
            (x, y)
            for y, row in enumerate(env.mdp.layout.rows)
            for x, cell in enumerate(row)
            if cell == terrain
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "Expected one {!r} station, got {}".format(
                    terrain, matches
                )
            )
        return matches[0]

    def move_next_to(target: Tuple[int, int]) -> None:
        start = env.state.players[0].position
        candidates = {
            Action.move_in_direction(target, direction)
            for direction in Action.MOTION_ACTIONS
            if direction != Action.STAY
            and env.mdp.layout.terrain_at(
                Action.move_in_direction(target, direction)
            )
            == FLOOR
        }
        queue = deque([(start, [])])
        visited = {start}
        selected_path: Optional[List[Tuple[int, int]]] = None
        while queue:
            position, path = queue.popleft()
            if position in candidates:
                selected_path = path
                break
            for direction in Action.MOTION_ACTIONS:
                if direction == Action.STAY:
                    continue
                next_position = Action.move_in_direction(position, direction)
                if (
                    next_position not in visited
                    and env.mdp.layout.terrain_at(next_position) == FLOOR
                ):
                    visited.add(next_position)
                    queue.append((next_position, path + [direction]))
        if selected_path is None:
            raise RuntimeError("Expert target is unreachable: {}".format(target))
        for direction in selected_path:
            submit(direction)

    def interact_at(target: Tuple[int, int]) -> None:
        move_next_to(target)
        submit(Action.PICK_DROP)

    plate_rack = terrain_position(PLATE_RACK)
    bun_dispenser = terrain_position(BUN_DISPENSER)
    lettuce_dispenser = terrain_position(LETTUCE_DISPENSER)
    beef_dispenser = terrain_position(BEEF_DISPENSER)
    grill = terrain_position(GRILL)
    serve = terrain_position(SERVE)
    interact_at(beef_dispenser)
    interact_at(grill)
    interact_at(plate_rack)
    interact_at(bun_dispenser)
    interact_at(lettuce_dispenser)
    move_next_to(grill)
    while env.state.grill.food == "raw_beef":
        submit(Action.STAY)
    interact_at(grill)
    interact_at(serve)

    if env.state.delivered_orders != 1:
        raise RuntimeError("Expert failed to complete one delivery")
    return ExpertDemo(
        observations=np.asarray(demo_observations, dtype=np.float32),
        available_actions=np.asarray(demo_available, dtype=np.float32),
        actions=np.asarray(demo_actions, dtype=np.int64),
        delivered_orders=env.state.delivered_orders,
    )


class BurgerVectorEnv:
    """Small synchronous vectorizer with no hidden process state."""

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        burger_config: BurgerConfig,
        start_stage: str = "standard",
        start_mix_stage: Optional[str] = None,
        randomize_start_positions: bool = False,
        random_start_max_objective_distance: Optional[int] = None,
        random_start_candidate_positions: Sequence[Tuple[int, int]] = (),
        training_episode_steps: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.training_episode_steps = training_episode_steps
        self.start_stages = tuple(
            start_mix_stage
            if start_mix_stage is not None and index % 2
            else start_stage
            for index in range(num_envs)
        )
        self.envs = [
            BurgerMAPPOEnv(
                mdp=BurgerGridworld(config=burger_config),
                num_players=num_agents,
                start_stage=self.start_stages[index],
                randomize_player_positions=randomize_start_positions,
                random_start_max_objective_distance=(
                    random_start_max_objective_distance
                ),
                random_start_candidate_positions=(
                    random_start_candidate_positions
                ),
            )
            for index in range(num_envs)
        ]
        resets = [
            env.reset(seed=seed + index)
            for index, env in enumerate(self.envs)
        ]
        self.observations = np.stack([item[0] for item in resets])
        self.shared_observations = np.stack([item[1] for item in resets])
        self.available_actions = np.stack([item[2] for item in resets])
        self.episode_returns = np.zeros(num_envs, dtype=np.float64)
        self.episode_steps = np.zeros(num_envs, dtype=np.int64)
        self.episode_initial_deliveries = np.asarray(
            [env._state_view.delivered_orders for env in self.envs],
            dtype=np.int64,
        )

    def step(
        self, action_batch: np.ndarray
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[List[Dict[str, object]]],
    ]:
        transitions = [
            env.step(actions)
            for env, actions in zip(self.envs, action_batch.tolist())
        ]
        observations = np.stack([item[0] for item in transitions])
        shared = np.stack([item[1] for item in transitions])
        available_actions = np.stack([item[5] for item in transitions])
        rewards = np.asarray(
            [item[2][0, 0] for item in transitions], dtype=np.float32
        )
        environment_dones = np.asarray(
            [item[3][0] for item in transitions], dtype=np.bool_
        )
        self.episode_steps += 1
        curriculum_truncations = np.asarray(
            [
                self.training_episode_steps is not None
                and stage != "standard"
                and self.episode_steps[index]
                >= self.training_episode_steps
                for index, stage in enumerate(self.start_stages)
            ],
            dtype=np.bool_,
        )
        dones = np.logical_or(environment_dones, curriculum_truncations)
        infos = [item[4] for item in transitions]

        # A shortened curriculum episode is an actual terminal boundary for
        # PPO, not an environment transition into the next reset state.
        # Replace gamma * Phi(s') with terminal Phi=0 so potential shaping
        # still telescopes and cannot be optimized through the time limit.
        for index, truncated in enumerate(curriculum_truncations):
            if truncated and not environment_dones[index]:
                mdp = self.envs[index].mdp
                terminal_phi = mdp.potential(
                    self.envs[index]._state_view
                )
                correction = (
                    -mdp.config.reward.potential_scale
                    * mdp.config.reward.gamma
                    * terminal_phi
                )
                rewards[index] += correction
                infos[index][0]["reward_breakdown"]["potential"] += (
                    correction
                )
                infos[index][0][
                    "training_terminal_potential_correction"
                ] = correction
        self.episode_returns += rewards

        for index, done in enumerate(dones):
            if done:
                infos[index][0]["training_truncated"] = bool(
                    curriculum_truncations[index]
                    and not environment_dones[index]
                )
                infos[index][0]["episode_deliveries"] = int(
                    self.envs[index]._state_view.delivered_orders
                    - self.episode_initial_deliveries[index]
                )
                reset_obs, reset_shared, reset_available = self.envs[index].reset()
                self.episode_initial_deliveries[index] = (
                    self.envs[index]._state_view.delivered_orders
                )
                self.episode_steps[index] = 0
                observations[index] = reset_obs
                shared[index] = reset_shared
                available_actions[index] = reset_available

        self.observations = observations
        self.shared_observations = shared
        self.available_actions = available_actions
        return observations, shared, rewards, dones, infos


class PPOTrainer:
    """Shared implementation: PPO for one actor, MAPPO for a team."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.preflight = audit_training_contract(config)
        _seed_everything(config.seed, config.deterministic_torch)
        self.device = _resolve_device(config.device)
        probe = BurgerMAPPOEnv(
            mdp=BurgerGridworld(config=environment_config(config)),
            num_players=config.num_agents,
        )
        observation_shape = probe.observation_space[0].shape
        shared_size = probe.share_observation_space[0].shape[0]
        self.actor = LocalActor(
            observation_shape,
            config.hidden_size,
            Action.NUM_ACTIONS,
        ).to(self.device)
        self.critic = CentralCritic(shared_size, config.hidden_size).to(
            self.device
        )
        if config.actor_init:
            self.load_actor(config.actor_init)
        self.reference_actor: Optional[LocalActor] = None
        if config.actor_anchor_kl_coef > 0:
            self.reference_actor = copy.deepcopy(self.actor).to(self.device)
            self.reference_actor.eval()
            for parameter in self.reference_actor.parameters():
                parameter.requires_grad_(False)
        if config.freeze_base_actor_for_emergency:
            for module in (
                self.actor.encoder,
                self.actor.agent_embedding,
                self.actor.policy,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        if config.freeze_cleanup_actor_for_fire:
            for module in (
                self.actor.cleanup_encoder,
                self.actor.cleanup_policy,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        if config.freeze_emergency_actor_for_suppression:
            for module in (
                self.actor.emergency_encoder,
                self.actor.emergency_policy,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        if config.freeze_suppression_actor_for_emergency:
            for module in (
                self.actor.suppression_encoder,
                self.actor.suppression_policy,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        if config.freeze_base_actor_for_dish:
            for module in (
                self.actor.encoder,
                self.actor.agent_embedding,
                self.actor.policy,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        if config.freeze_fire_actors_for_dish:
            for module in (
                self.actor.emergency_encoder,
                self.actor.emergency_policy,
                self.actor.cleanup_encoder,
                self.actor.cleanup_policy,
                self.actor.suppression_encoder,
                self.actor.suppression_policy,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        if config.freeze_dish_actor_for_assembly:
            for module in (
                self.actor.dish_encoder,
                self.actor.dish_policy,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        if config.freeze_assembly_actor_for_workflow:
            for module in (
                self.actor.assembly_encoder,
                self.actor.assembly_policy,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        actor_parameters = [
            parameter
            for parameter in self.actor.parameters()
            if parameter.requires_grad
        ]
        self.optimizer = torch.optim.Adam(
            actor_parameters + list(self.critic.parameters()),
            lr=config.learning_rate,
            eps=1e-5,
        )
        self.vector_env = BurgerVectorEnv(
            config.num_envs,
            config.num_agents,
            environment_config(config),
            start_stage=config.training_start_stage,
            start_mix_stage=config.training_start_mix_stage,
            randomize_start_positions=config.randomize_start_positions,
            random_start_max_objective_distance=(
                config.random_start_max_objective_distance
            ),
            random_start_candidate_positions=(
                config.random_start_candidate_positions
            ),
            training_episode_steps=config.training_episode_steps,
            seed=config.seed,
        )
        self.expert_demo = (
            generate_single_agent_expert_demo(config)
            if config.bc_pretrain_steps > 0
            else None
        )
        self.agent_ids = torch.arange(
            config.num_agents, device=self.device, dtype=torch.long
        ).repeat(config.num_envs)
        self.global_step = 0
        self.update = 0
        self.action_freeze_streak = 0

    def _behavior_cloning_loss(self, batch_size: int) -> torch.Tensor:
        if self.expert_demo is None:
            raise RuntimeError("Behavior-cloning data is not configured")
        sample_count = len(self.expert_demo.actions)
        indices = torch.randint(
            0,
            sample_count,
            (batch_size,),
            device=self.device,
        )
        observations = torch.as_tensor(
            self.expert_demo.observations,
            dtype=torch.float32,
            device=self.device,
        )[indices]
        available = torch.as_tensor(
            self.expert_demo.available_actions,
            dtype=torch.float32,
            device=self.device,
        )[indices]
        actions = torch.as_tensor(
            self.expert_demo.actions,
            dtype=torch.long,
            device=self.device,
        )[indices]
        agent_ids = torch.zeros(
            batch_size, dtype=torch.long, device=self.device
        )
        logits = _masked_logits(
            self.actor(observations, agent_ids),
            available,
        )
        return F.cross_entropy(logits, actions)

    def pretrain_actor(self) -> Dict[str, float]:
        """Warm-start the local actor on legal environment demonstrations."""

        if self.config.bc_pretrain_steps == 0:
            return {
                "bc_steps": 0.0,
                "bc_demo_transitions": 0.0,
                "bc_initial_loss": float("nan"),
                "bc_final_loss": float("nan"),
            }
        if self.expert_demo is None:
            raise RuntimeError("Missing behavior-cloning demonstration")
        optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config.bc_learning_rate,
            eps=1e-5,
        )
        losses: List[float] = []
        self.actor.train()
        for _ in range(self.config.bc_pretrain_steps):
            loss = self._behavior_cloning_loss(
                min(
                    self.config.bc_batch_size,
                    len(self.expert_demo.actions),
                )
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.config.max_grad_norm,
                error_if_nonfinite=True,
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        return {
            "bc_steps": float(self.config.bc_pretrain_steps),
            "bc_demo_transitions": float(len(self.expert_demo.actions)),
            "bc_initial_loss": float(losses[0]),
            "bc_final_loss": float(np.mean(losses[-min(50, len(losses)) :])),
        }

    def load_actor(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        state_dict = checkpoint.get("actor", checkpoint)
        missing, unexpected = load_actor_state_dict_compatible(
            self.actor, state_dict, strict=False
        )
        allowed_missing = {
            "agent_embedding.weight",
            *{
                name
                for name in self.actor.state_dict()
                if name.startswith(
                    (
                        "emergency_",
                        "cleanup_",
                        "suppression_",
                        "dish_",
                        "assembly_",
                        "workflow_",
                    )
                )
            },
        }
        if unexpected or set(missing) - allowed_missing:
            raise ValueError(
                "Actor checkpoint mismatch: missing={}, unexpected={}".format(
                    missing, unexpected
                )
            )
        train_config = checkpoint.get("train_config", {})
        source_agents = (
            train_config.get("num_agents")
            if isinstance(train_config, dict)
            else None
        )
        if self.config.num_agents > 1 and not (
            isinstance(source_agents, int)
            and 1 <= source_agents <= MAX_AGENTS
        ):
            raise ValueError(
                "MAPPO actor initialization requires checkpoint "
                "train_config.num_agents"
            )
        if (
            isinstance(source_agents, int)
            and 1 <= source_agents < self.config.num_agents
        ):
            with torch.no_grad():
                embeddings = self.actor.agent_embedding.weight
                trained_mean = embeddings[:source_agents].mean(
                    dim=0, keepdim=True
                )
                embeddings[source_agents:].copy_(
                    trained_mean.expand_as(embeddings[source_agents:])
                )

    @torch.no_grad()
    def collect_rollout(self) -> Rollout:
        config = self.config
        observations: List[torch.Tensor] = []
        shared_observations: List[torch.Tensor] = []
        available_actions: List[torch.Tensor] = []
        actions: List[torch.Tensor] = []
        log_probs: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        rewards: List[torch.Tensor] = []
        dones: List[torch.Tensor] = []
        completed_returns: List[float] = []
        completed_deliveries: List[int] = []
        collisions = 0
        fires = 0
        action_counts = np.zeros(Action.NUM_ACTIONS, dtype=np.int64)
        all_agents_stay_steps = 0
        all_agents_noop_steps = 0
        joint_steps = 0
        legal_action_slots = 0
        actor_slots = 0
        reward_component_totals = {
            "correct_delivery": 0.0,
            "raw_beef_placed": 0.0,
            "dirty_plate_pickup": 0.0,
            "wash_started": 0.0,
            "wash_progress": 0.0,
            "plate_washed": 0.0,
            "fire_extinguished": 0.0,
            "fire_started": 0.0,
            "fire_active": 0.0,
            "fire_food_handling": 0.0,
            "dirty_plate_counter_handling": 0.0,
            "collision": 0.0,
            "time_step": 0.0,
            "potential": 0.0,
        }
        collect_started = time.perf_counter()
        stay_index = Action.ACTION_TO_INDEX[Action.STAY]

        for _ in range(config.rollout_length):
            local = torch.as_tensor(
                self.vector_env.observations[:, : config.num_agents],
                dtype=torch.float32,
                device=self.device,
            )
            shared = torch.as_tensor(
                self.vector_env.shared_observations[:, 0],
                dtype=torch.float32,
                device=self.device,
            )
            flat_local = local.flatten(0, 1)
            available = torch.as_tensor(
                self.vector_env.available_actions[
                    :, : config.num_agents
                ],
                dtype=torch.float32,
                device=self.device,
            )
            logits = _masked_logits(
                self.actor(flat_local, self.agent_ids),
                available.flatten(0, 1),
            ) / config.rollout_temperature
            distribution = Categorical(logits=logits)
            active_actions = distribution.sample()
            active_log_probs = distribution.log_prob(active_actions)
            team_values = self.critic(shared)
            active_action_matrix = active_actions.view(
                config.num_envs, config.num_agents
            )
            action_counts += np.bincount(
                active_action_matrix.cpu().numpy().reshape(-1),
                minlength=Action.NUM_ACTIONS,
            )
            all_agents_stay_steps += int(
                torch.all(
                    active_action_matrix == stay_index, dim=1
                ).sum()
            )
            joint_steps += config.num_envs
            legal_action_slots += int(available.sum().item())
            actor_slots += config.num_envs * config.num_agents
            old_positions = [
                tuple(
                    player.position
                    for player in vector_env._state_view.players
                )
                for vector_env in self.vector_env.envs
            ]

            joint_actions = np.full(
                (config.num_envs, MAX_AGENTS),
                stay_index,
                dtype=np.int64,
            )
            joint_actions[:, : config.num_agents] = (
                active_action_matrix
                .cpu()
                .numpy()
            )

            observations.append(local.cpu())
            shared_observations.append(shared.cpu())
            available_actions.append(available.cpu())
            actions.append(
                active_actions.view(config.num_envs, config.num_agents).cpu()
            )
            log_probs.append(
                active_log_probs.view(config.num_envs, config.num_agents).cpu()
            )
            values.append(team_values.cpu())

            _, _, reward_batch, done_batch, info_batch = self.vector_env.step(
                joint_actions
            )
            rewards.append(torch.from_numpy(reward_batch.copy()))
            dones.append(torch.from_numpy(done_batch.copy()))
            self.global_step += config.num_envs

            for env_index, (done, infos) in enumerate(
                zip(done_batch, info_batch)
            ):
                event_types = [
                    str(event["type"])
                    for event in infos[0].get("events", ())
                ]
                new_positions = tuple(
                    player.position
                    for player in self.vector_env.envs[
                        env_index
                    ]._state_view.players
                )
                agent_event = any(
                    "agent" in event or "agents" in event
                    for event in infos[0].get("events", ())
                )
                if (
                    not done
                    and new_positions == old_positions[env_index]
                    and not agent_event
                ):
                    all_agents_noop_steps += 1
                collisions += event_types.count("collision")
                fires += event_types.count("fire_started")
                breakdown = infos[0]["reward_breakdown"]
                if not math.isclose(
                    float(reward_batch[env_index]),
                    sum(float(value) for value in breakdown.values()),
                    rel_tol=0.0,
                    abs_tol=1e-5,
                ):
                    raise RuntimeError(
                        "Rollout reward differs from environment breakdown"
                    )
                for name in reward_component_totals:
                    reward_component_totals[name] += float(breakdown[name])
                if done:
                    completed_returns.append(
                        float(self.vector_env.episode_returns[env_index])
                    )
                    completed_deliveries.append(
                        int(infos[0]["episode_deliveries"])
                    )
                    self.vector_env.episode_returns[env_index] = 0.0

        next_shared = torch.as_tensor(
            self.vector_env.shared_observations[:, 0],
            dtype=torch.float32,
            device=self.device,
        )
        next_values = self.critic(next_shared).cpu()
        reward_tensor = torch.stack(rewards)
        done_tensor = torch.stack(dones)
        value_tensor = torch.stack(values)
        advantages = torch.zeros_like(reward_tensor)
        last_gae = torch.zeros(config.num_envs)
        for step in reversed(range(config.rollout_length)):
            if step == config.rollout_length - 1:
                next_value = next_values
            else:
                next_value = value_tensor[step + 1]
            not_terminal = 1.0 - done_tensor[step].float()
            delta = (
                reward_tensor[step]
                + config.gamma * next_value * not_terminal
                - value_tensor[step]
            )
            last_gae = (
                delta
                + config.gamma
                * config.gae_lambda
                * not_terminal
                * last_gae
            )
            advantages[step] = last_gae

        return Rollout(
            observations=torch.stack(observations),
            shared_observations=torch.stack(shared_observations),
            available_actions=torch.stack(available_actions),
            actions=torch.stack(actions),
            log_probs=torch.stack(log_probs),
            values=value_tensor,
            rewards=reward_tensor,
            dones=done_tensor,
            advantages=advantages,
            returns=advantages + value_tensor,
            completed_returns=completed_returns,
            completed_deliveries=completed_deliveries,
            collision_events=collisions,
            fire_events=fires,
            action_counts=tuple(int(value) for value in action_counts),
            all_agents_stay_steps=all_agents_stay_steps,
            all_agents_noop_steps=all_agents_noop_steps,
            joint_steps=joint_steps,
            legal_action_slots=legal_action_slots,
            actor_slots=actor_slots,
            reward_component_totals=reward_component_totals,
            environment_seconds=time.perf_counter() - collect_started,
        )

    def update_policy(self, rollout: Rollout) -> Dict[str, float]:
        config = self.config
        rollout_health = self._rollout_health_metrics(rollout)
        transition_count = config.rollout_length * config.num_envs
        if transition_count < config.num_minibatches:
            raise ValueError("num_minibatches exceeds rollout transitions")
        batch_size = transition_count // config.num_minibatches

        observations = rollout.observations.reshape(
            transition_count,
            config.num_agents,
            *rollout.observations.shape[3:],
        )
        available_actions = rollout.available_actions.reshape(
            transition_count,
            config.num_agents,
            Action.NUM_ACTIONS,
        )
        shared = rollout.shared_observations.reshape(transition_count, -1)
        actions = rollout.actions.reshape(
            transition_count, config.num_agents
        )
        old_log_probs = rollout.log_probs.reshape(
            transition_count, config.num_agents
        )
        old_values = rollout.values.reshape(transition_count)
        returns = rollout.returns.reshape(transition_count)
        advantages = rollout.advantages.reshape(transition_count)
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

        metrics: Dict[str, List[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "bc_loss": [],
            "actor_anchor_kl": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "grad_norm": [],
        }
        for _ in range(config.update_epochs):
            permutation = torch.randperm(transition_count)
            for start in range(0, transition_count, batch_size):
                indices = permutation[start : start + batch_size]
                count = len(indices)
                local_batch = observations[indices].to(self.device)
                agent_ids = torch.arange(
                    config.num_agents,
                    device=self.device,
                    dtype=torch.long,
                ).repeat(count)
                action_mask = (
                    available_actions[indices]
                    .to(self.device)
                    .flatten(0, 1)
                )
                flat_local_batch = local_batch.flatten(0, 1)
                raw_logits = _masked_logits(
                    self.actor(flat_local_batch, agent_ids),
                    action_mask,
                )
                logits = raw_logits / config.rollout_temperature
                distribution = Categorical(logits=logits)
                action_batch = actions[indices].to(self.device).flatten()
                new_log_prob = distribution.log_prob(action_batch).view(
                    count, config.num_agents
                )
                entropy = distribution.entropy().mean()
                old_log_prob = old_log_probs[indices].to(self.device)
                log_ratio = new_log_prob - old_log_prob
                ratio = log_ratio.exp()
                advantage = advantages[indices].to(self.device).unsqueeze(-1)
                policy_loss = -torch.min(
                    ratio * advantage,
                    torch.clamp(
                        ratio,
                        1.0 - config.clip_coef,
                        1.0 + config.clip_coef,
                    )
                    * advantage,
                ).mean()

                new_value = self.critic(shared[indices].to(self.device))
                old_value = old_values[indices].to(self.device)
                return_batch = returns[indices].to(self.device)
                value_unclipped = (new_value - return_batch).pow(2)
                value_clipped = old_value + torch.clamp(
                    new_value - old_value,
                    -config.clip_coef,
                    config.clip_coef,
                )
                value_loss = 0.5 * torch.max(
                    value_unclipped,
                    (value_clipped - return_batch).pow(2),
                ).mean()
                bc_loss = (
                    self._behavior_cloning_loss(config.bc_batch_size)
                    if config.bc_aux_coef > 0
                    else torch.zeros((), device=self.device)
                )
                if self.reference_actor is not None:
                    with torch.no_grad():
                        reference_logits = _masked_logits(
                            self.reference_actor(
                                flat_local_batch, agent_ids
                            ),
                            action_mask,
                        )
                    actor_anchor_kl = torch.distributions.kl_divergence(
                        Categorical(logits=reference_logits),
                        Categorical(logits=raw_logits),
                    ).mean()
                else:
                    actor_anchor_kl = torch.zeros((), device=self.device)
                loss = (
                    policy_loss
                    + config.value_coef * value_loss
                    - config.entropy_coef * entropy
                    + config.bc_aux_coef * bc_loss
                    + config.actor_anchor_kl_coef * actor_anchor_kl
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    list(self.actor.parameters())
                    + list(self.critic.parameters()),
                    config.max_grad_norm,
                    error_if_nonfinite=True,
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs() > config.clip_coef
                    ).float().mean()
                metrics["policy_loss"].append(float(policy_loss.detach()))
                metrics["value_loss"].append(float(value_loss.detach()))
                metrics["bc_loss"].append(float(bc_loss.detach()))
                metrics["actor_anchor_kl"].append(
                    float(actor_anchor_kl.detach())
                )
                metrics["entropy"].append(float(entropy.detach()))
                metrics["approx_kl"].append(float(approx_kl.detach()))
                metrics["clip_fraction"].append(float(clip_fraction.detach()))
                metrics["grad_norm"].append(float(grad_norm.detach()))

        return {
            name: float(np.mean(values)) for name, values in metrics.items()
        } | rollout_health

    def _rollout_health_metrics(
        self, rollout: Rollout
    ) -> Dict[str, float]:
        frozen = (
            rollout.all_agents_stay_ratio
            >= self.config.max_all_agents_stay_ratio
            or rollout.all_agents_noop_ratio
            >= self.config.max_all_agents_noop_ratio
        )
        self.action_freeze_streak = (
            self.action_freeze_streak + 1 if frozen else 0
        )
        if (
            self.action_freeze_streak
            >= self.config.action_freeze_patience_updates
        ):
            raise RuntimeError(
                "Action-freeze gate failed for {} consecutive updates: "
                "all_stay={:.4f}, all_noop={:.4f}".format(
                    self.action_freeze_streak,
                    rollout.all_agents_stay_ratio,
                    rollout.all_agents_noop_ratio,
                )
            )

        total_actions = max(sum(rollout.action_counts), 1)
        names = (
            "north",
            "south",
            "east",
            "west",
            "stay",
            "pick_drop",
            "process",
        )
        result = {
            "rollout_all_agents_stay_ratio": rollout.all_agents_stay_ratio,
            "rollout_all_agents_noop_ratio": rollout.all_agents_noop_ratio,
            "rollout_mean_legal_actions": rollout.mean_legal_actions,
            "action_freeze_streak": float(self.action_freeze_streak),
        }
        result.update(
            {
                "action_fraction_{}".format(name): count / total_actions
                for name, count in zip(names, rollout.action_counts)
            }
        )
        result.update(
            {
                "reward_component_{}".format(name): float(value)
                for name, value in rollout.reward_component_totals.items()
            }
        )
        return result

    def save_checkpoint(self, output_dir: Path, label: str) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / "{}.pt".format(label)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "train_config": asdict(self.config),
                "global_step": self.global_step,
                "update": self.update,
            },
            checkpoint_path,
        )
        return checkpoint_path


@torch.no_grad()
def evaluate_policy(
    actor: LocalActor,
    config: TrainConfig,
    device: torch.device,
    episodes: Optional[int] = None,
    start_stage: str = "standard",
    random_start_max_objective_distance: Optional[int] = None,
    random_start_candidate_positions: Optional[Sequence[Point]] = None,
) -> Dict[str, float]:
    """Deterministic evaluation; sparse success is reported separately."""

    episode_count = episodes or config.eval_episodes
    total_rewards: List[float] = []
    sparse_rewards: List[float] = []
    deliveries: List[int] = []
    collisions: List[int] = []
    fires: List[int] = []
    raw_beef_placements: List[int] = []
    washed_plates: List[int] = []
    dirty_plate_pickups: List[int] = []
    fires_extinguished: List[int] = []
    fire_active_steps: List[int] = []
    discarded_items: List[int] = []
    longest_stagnation_steps: List[int] = []
    action_counts = np.zeros(Action.NUM_ACTIONS, dtype=np.int64)
    all_agents_stay_steps = 0
    all_agents_noop_steps = 0
    evaluation_steps = 0
    stay_index = Action.ACTION_TO_INDEX[Action.STAY]
    actor.eval()

    for episode in range(episode_count):
        env = BurgerMAPPOEnv(
            mdp=BurgerGridworld(config=environment_config(config)),
            num_players=config.num_agents,
            start_stage=start_stage,
            randomize_player_positions=config.randomize_start_positions,
            random_start_max_objective_distance=(
                random_start_max_objective_distance
            ),
            random_start_candidate_positions=(
                config.random_start_candidate_positions
                if random_start_candidate_positions is None
                else random_start_candidate_positions
            ),
        )
        observations, _, available = env.reset(seed=config.seed + episode)
        initial_deliveries = env.state.delivered_orders
        episode_reward = 0.0
        episode_sparse = 0.0
        episode_collisions = 0
        episode_fires = 0
        episode_raw_beef_placements = 0
        episode_washed_plates = 0
        episode_dirty_plate_pickups = 0
        episode_fires_extinguished = 0
        episode_fire_active_steps = 0
        episode_discarded_items = 0
        episode_stagnation = 0
        episode_longest_stagnation = 0
        done = False
        while not done:
            local = torch.as_tensor(
                observations[: config.num_agents],
                dtype=torch.float32,
                device=device,
            )
            agent_ids = torch.arange(
                config.num_agents, device=device, dtype=torch.long
            )
            action_mask = torch.as_tensor(
                available[: config.num_agents],
                dtype=torch.float32,
                device=device,
            )
            selected = (
                _masked_logits(actor(local, agent_ids), action_mask)
                .argmax(dim=-1)
                .cpu()
                .numpy()
            )
            action_counts += np.bincount(
                selected, minlength=Action.NUM_ACTIONS
            )
            all_agents_stay_steps += int(
                np.all(selected == stay_index)
            )
            evaluation_steps += 1
            old_positions = tuple(
                player.position for player in env.state.players
            )
            joint_actions = np.full(MAX_AGENTS, stay_index, dtype=np.int64)
            joint_actions[: config.num_agents] = selected
            (
                observations,
                _,
                reward,
                dones,
                infos,
                available,
            ) = env.step(joint_actions)
            episode_reward += float(reward[0, 0])
            breakdown = infos[0]["reward_breakdown"]
            episode_sparse += float(breakdown["correct_delivery"])
            event_types = [
                str(event["type"]) for event in infos[0].get("events", ())
            ]
            new_positions = tuple(
                player.position for player in env.state.players
            )
            agent_event = any(
                "agent" in event or "agents" in event
                for event in infos[0].get("events", ())
            )
            if new_positions == old_positions and not agent_event:
                all_agents_noop_steps += 1
            task_progress = any(
                event_type
                in {
                    "correct_delivery",
                    "raw_beef_placed",
                    "cooked_beef_added_to_held_plate",
                    "ingredient_added_from_dispenser",
                    "ingredient_added_to_held_plate",
                    "ingredient_added_to_counter_plate",
                    "dirty_plate_pickup",
                    "wash_started",
                    "wash_progress",
                    "plate_washed",
                    "fire_extinguished",
                }
                for event_type in event_types
            )
            if new_positions != old_positions or task_progress:
                episode_stagnation = 0
            else:
                episode_stagnation += 1
                episode_longest_stagnation = max(
                    episode_longest_stagnation,
                    episode_stagnation,
                )
            episode_collisions += event_types.count("collision")
            episode_fires += event_types.count("fire_started")
            episode_fires_extinguished += event_types.count(
                "fire_extinguished"
            )
            episode_fire_active_steps += event_types.count("fire_active")
            episode_raw_beef_placements += event_types.count(
                "raw_beef_placed"
            )
            episode_washed_plates += event_types.count("plate_washed")
            episode_dirty_plate_pickups += event_types.count(
                "dirty_plate_pickup"
            )
            episode_discarded_items += (
                event_types.count("food_discarded")
                + event_types.count("plate_contents_discarded")
            )
            done = bool(dones[0])
        total_rewards.append(episode_reward)
        sparse_rewards.append(episode_sparse)
        deliveries.append(
            env.state.delivered_orders - initial_deliveries
        )
        collisions.append(episode_collisions)
        fires.append(episode_fires)
        raw_beef_placements.append(episode_raw_beef_placements)
        washed_plates.append(episode_washed_plates)
        dirty_plate_pickups.append(episode_dirty_plate_pickups)
        fires_extinguished.append(episode_fires_extinguished)
        fire_active_steps.append(episode_fire_active_steps)
        discarded_items.append(episode_discarded_items)
        longest_stagnation_steps.append(episode_longest_stagnation)

    actor.train()
    return {
        "eval_mean_reward": float(np.mean(total_rewards)),
        "eval_mean_sparse_reward": float(np.mean(sparse_rewards)),
        "eval_mean_deliveries": float(np.mean(deliveries)),
        "eval_deliveries_per_minute": float(
            np.mean(deliveries) / (config.horizon * 0.42 / 60.0)
        ),
        "eval_mean_collisions": float(np.mean(collisions)),
        "eval_mean_fires": float(np.mean(fires)),
        "eval_mean_fires_extinguished": float(
            np.mean(fires_extinguished)
        ),
        "eval_fire_extinguish_success_rate": float(
            np.mean(np.asarray(fires_extinguished) >= 1)
        ),
        "eval_mean_fire_active_steps": float(np.mean(fire_active_steps)),
        "eval_max_fire_active_steps": float(np.max(fire_active_steps)),
        "eval_mean_raw_beef_placements": float(
            np.mean(raw_beef_placements)
        ),
        "eval_raw_beef_placement_rate": float(
            np.mean(np.asarray(raw_beef_placements) >= 1)
        ),
        "eval_mean_washed_plates": float(np.mean(washed_plates)),
        "eval_wash_success_rate": float(
            np.mean(np.asarray(washed_plates) >= 1)
        ),
        "eval_dirty_plate_pickup_rate": float(
            np.mean(np.asarray(dirty_plate_pickups) >= 1)
        ),
        "eval_second_delivery_rate": float(
            np.mean(np.asarray(deliveries) >= 2)
        ),
        "eval_delivery_success_rate": float(
            np.mean(np.asarray(deliveries) >= 1)
        ),
        "eval_mean_discarded_items": float(np.mean(discarded_items)),
        "eval_mean_longest_stagnation_steps": float(
            np.mean(longest_stagnation_steps)
        ),
        "eval_max_longest_stagnation_steps": float(
            np.max(longest_stagnation_steps)
        ),
        "eval_stay_action_ratio": float(
            action_counts[stay_index] / max(action_counts.sum(), 1)
        ),
        "eval_all_agents_stay_ratio": float(
            all_agents_stay_steps / max(evaluation_steps, 1)
        ),
        "eval_all_agents_noop_ratio": float(
            all_agents_noop_steps / max(evaluation_steps, 1)
        ),
        "eval_action_coverage": float(
            np.count_nonzero(action_counts) / Action.NUM_ACTIONS
        ),
    }


def evaluate_configured_stages(
    actor: LocalActor,
    config: TrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Report the standard objective and every configured curriculum reset."""

    # Failure-focused candidate pools belong only to the curriculum stage.
    # The standard regression gate must continue to cover the full map.
    metrics = evaluate_policy(
        actor,
        config,
        device,
        start_stage="standard",
        random_start_candidate_positions=(),
    )
    if config.training_start_stage != "standard":
        stage_metrics = evaluate_policy(
            actor,
            config,
            device,
            start_stage=config.training_start_stage,
        )
        metrics.update(
            {
                "train_stage_{}".format(name): value
                for name, value in stage_metrics.items()
            }
        )
        if config.random_start_max_objective_distance is not None:
            curriculum_metrics = evaluate_policy(
                actor,
                config,
                device,
                start_stage=config.training_start_stage,
                random_start_max_objective_distance=(
                    config.random_start_max_objective_distance
                ),
            )
            metrics.update(
                {
                    "curriculum_train_stage_{}".format(name): value
                    for name, value in curriculum_metrics.items()
                }
            )
    if (
        config.training_start_mix_stage is not None
        and config.training_start_mix_stage
        not in {"standard", config.training_start_stage}
    ):
        mix_metrics = evaluate_policy(
            actor,
            config,
            device,
            start_stage=config.training_start_mix_stage,
        )
        metrics.update(
            {
                "mix_stage_{}".format(name): value
                for name, value in mix_metrics.items()
            }
        )
        if (
            config.random_start_max_objective_distance is not None
            and config.training_start_mix_stage != "standard"
        ):
            curriculum_mix_metrics = evaluate_policy(
                actor,
                config,
                device,
                start_stage=config.training_start_mix_stage,
                random_start_max_objective_distance=(
                    config.random_start_max_objective_distance
                ),
            )
            metrics.update(
                {
                    "curriculum_mix_stage_{}".format(name): value
                    for name, value in curriculum_mix_metrics.items()
                }
            )
    return metrics


def train(config: TrainConfig) -> Path:
    run_name = "{}_{}agent_v2_seed{}".format(
        config.algorithm, config.num_agents, config.seed
    )
    output_dir = Path(config.output_dir) / run_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            "Refusing to overwrite existing training run: {}. "
            "Choose a new --output-dir.".format(output_dir)
        )

    trainer = PPOTrainer(config)
    pretraining = (
        trainer.pretrain_actor()
        if config.bc_pretrain_steps > 0
        else {"bc_steps": 0.0}
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "preflight.json").write_text(
        json.dumps(trainer.preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "pretraining.json").write_text(
        json.dumps(pretraining, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if config.bc_pretrain_steps > 0:
        trainer.save_checkpoint(output_dir, "pretrained")
    metrics_path = output_dir / "metrics.csv"
    updates = math.ceil(
        config.total_env_steps / (config.num_envs * config.rollout_length)
    )
    training_started = time.perf_counter()
    fieldnames: Optional[List[str]] = None

    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        writer: Optional[csv.DictWriter] = None
        for update in range(1, updates + 1):
            trainer.update = update
            rollout = trainer.collect_rollout()
            update_started = time.perf_counter()
            optimization = trainer.update_policy(rollout)
            optimize_seconds = time.perf_counter() - update_started
            elapsed = time.perf_counter() - training_started
            steps_per_second = trainer.global_step / max(elapsed, 1e-9)
            row: Dict[str, float] = {
                "update": float(update),
                "global_env_step": float(trainer.global_step),
                "steps_per_second": steps_per_second,
                "collect_seconds": rollout.environment_seconds,
                "optimize_seconds": optimize_seconds,
                "train_mean_reward": (
                    float(np.mean(rollout.completed_returns))
                    if rollout.completed_returns
                    else float("nan")
                ),
                "train_mean_deliveries": (
                    float(np.mean(rollout.completed_deliveries))
                    if rollout.completed_deliveries
                    else float("nan")
                ),
                "rollout_collisions": float(rollout.collision_events),
                "rollout_fires": float(rollout.fire_events),
                **optimization,
            }
            should_evaluate = (
                update == 1
                or update == updates
                or update % config.eval_interval_updates == 0
            )
            if should_evaluate:
                row.update(
                    evaluate_configured_stages(
                        trainer.actor,
                        config,
                        trainer.device,
                    )
                )
            else:
                row.update(
                    {
                        "eval_mean_reward": float("nan"),
                        "eval_mean_sparse_reward": float("nan"),
                        "eval_mean_deliveries": float("nan"),
                        "eval_deliveries_per_minute": float("nan"),
                        "eval_mean_collisions": float("nan"),
                        "eval_mean_fires": float("nan"),
                        "eval_stay_action_ratio": float("nan"),
                        "eval_all_agents_stay_ratio": float("nan"),
                        "eval_all_agents_noop_ratio": float("nan"),
                        "eval_action_coverage": float("nan"),
                    }
                )

            if writer is None:
                fieldnames = list(row)
                writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
                writer.writeheader()
            writer.writerow(row)
            metrics_file.flush()
            print(json.dumps(row, sort_keys=True), flush=True)

            if (
                update == updates
                or update % config.checkpoint_interval_updates == 0
            ):
                trainer.save_checkpoint(
                    output_dir, "checkpoint_{:06d}".format(update)
                )

    final_path = trainer.save_checkpoint(output_dir, "final")
    summary = {
        "checkpoint": str(final_path),
        "preflight": trainer.preflight,
        "pretraining": pretraining,
        "elapsed_seconds": time.perf_counter() - training_started,
        "global_env_steps": trainer.global_step,
        "measured_steps_per_second": trainer.global_step
        / max(time.perf_counter() - training_started, 1e-9),
        "final_evaluation": evaluate_configured_stages(
            trainer.actor, config, trainer.device
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return final_path


def _seed_everything(seed: int, deterministic_torch: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _parse_start_position(value: str) -> Tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "start position must use X,Y"
        )
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "start position coordinates must be integers"
        ) from error


def parse_args(argv: Optional[Sequence[str]] = None) -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train one-agent PPO or multi-agent MAPPO on one burger map"
    )
    parser.add_argument("--algorithm", choices=("ppo", "mappo"), required=True)
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--total-env-steps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-length", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument(
        "--actor-anchor-kl-coef",
        type=float,
        default=0.0,
        help=(
            "KL penalty to the actor-init policy on the same observations. "
            "Use it for continual hard-case training without forgetting an "
            "already accepted checkpoint."
        ),
    )
    parser.add_argument(
        "--rollout-temperature",
        type=float,
        default=1.0,
        help=(
            "Temperature applied consistently to the PPO behavior and update "
            "distributions. Evaluation remains deterministic at temperature 1."
        ),
    )
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=429)
    parser.add_argument("--cook-steps", type=int, default=24)
    parser.add_argument("--burn-steps", type=int, default=16)
    parser.add_argument("--wash-steps", type=int, default=10)
    parser.add_argument("--plate-return-steps", type=int, default=12)
    parser.add_argument("--correct-delivery-reward", type=float, default=20.0)
    parser.add_argument("--raw-beef-placed-reward", type=float, default=0.0)
    parser.add_argument("--dirty-plate-pickup-reward", type=float, default=0.0)
    parser.add_argument("--wash-started-reward", type=float, default=0.0)
    parser.add_argument("--wash-progress-reward", type=float, default=0.0)
    parser.add_argument("--plate-washed-reward", type=float, default=0.0)
    parser.add_argument(
        "--fire-extinguished-reward", type=float, default=0.0
    )
    parser.add_argument("--fire-started-penalty", type=float, default=-5.0)
    parser.add_argument(
        "--fire-active-penalty",
        type=float,
        default=-0.25,
        help=(
            "Safety cost for every transition that ends with an active grill "
            "fire. This makes earlier extinguishing strictly better without "
            "creating a repeatable extinguish bonus."
        ),
    )
    parser.add_argument(
        "--fire-food-handling-penalty",
        type=float,
        default=-2.0,
        help=(
            "One-time safety cost for touching ingredient dispensers while "
            "the grill is on fire; the action remains gameplay-legal."
        ),
    )
    parser.add_argument(
        "--dirty-plate-counter-handling-penalty",
        type=float,
        default=-0.25,
        help=(
            "Small cost for placing or retrieving a dirty plate on a "
            "worktop; the action remains legal."
        ),
    )
    parser.add_argument("--collision-penalty", type=float, default=-0.05)
    parser.add_argument("--time-step-penalty", type=float, default=0.0)
    parser.add_argument("--potential-scale", type=float, default=1.0)
    parser.add_argument(
        "--navigation-potential-scale", type=float, default=1.0
    )
    parser.add_argument(
        "--max-all-agents-stay-ratio", type=float, default=0.98
    )
    parser.add_argument(
        "--max-all-agents-noop-ratio", type=float, default=0.995
    )
    parser.add_argument(
        "--action-freeze-patience-updates", type=int, default=3
    )
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-interval-updates", type=int, default=20)
    parser.add_argument("--checkpoint-interval-updates", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="runs/burger")
    parser.add_argument("--actor-init")
    parser.add_argument("--deterministic-torch", action="store_true")
    parser.add_argument(
        "--training-start-stage",
        choices=CURRICULUM_START_STAGES,
        default="standard",
    )
    parser.add_argument(
        "--training-start-mix-stage",
        choices=CURRICULUM_START_STAGES,
    )
    parser.add_argument(
        "--randomize-start-positions",
        action="store_true",
    )
    parser.add_argument(
        "--random-start-max-objective-distance",
        type=int,
        help=(
            "For hard one-agent starts, sample initial positions no farther "
            "than this shortest-path distance from the recovery station. "
            "Evaluation remains full-map."
        ),
    )
    parser.add_argument(
        "--random-start-candidate-position",
        action="append",
        type=_parse_start_position,
        dest="random_start_candidate_positions",
        default=[],
        metavar="X,Y",
        help=(
            "Restrict randomized training starts to a walkable coordinate. "
            "Repeat this flag to define a failure-focused candidate pool."
        ),
    )
    parser.add_argument(
        "--training-episode-steps",
        type=int,
        help=(
            "Reset curriculum training episodes after this many control "
            "steps. The authoritative environment and evaluation horizon "
            "remain unchanged."
        ),
    )
    parser.add_argument(
        "--allow-mixed-curriculum-event-rewards",
        action="store_true",
        help=(
            "Explicitly allow temporary fire-extinguish reward in a "
            "standard-plus-fire curriculum. Final consolidation must omit it."
        ),
    )
    parser.add_argument(
        "--freeze-base-actor-for-emergency",
        action="store_true",
        help=(
            "Freeze the standard actor and train only the locally gated "
            "emergency residual branch."
        ),
    )
    parser.add_argument(
        "--freeze-cleanup-actor-for-fire",
        action="store_true",
        help=(
            "Freeze the post-fire extinguisher cleanup residual while "
            "training the visible-fire response branch."
        ),
    )
    parser.add_argument(
        "--freeze-emergency-actor-for-suppression",
        action="store_true",
        help=(
            "Freeze the plate/empty-hand emergency residual while training "
            "the held-extinguisher suppression branch."
        ),
    )
    parser.add_argument(
        "--freeze-suppression-actor-for-emergency",
        action="store_true",
        help=(
            "Freeze the held-extinguisher suppression residual while "
            "training the plate/empty-hand emergency branch."
        ),
    )
    parser.add_argument(
        "--freeze-base-actor-for-dish",
        action="store_true",
        help=(
            "Freeze the standard actor and train only the plate-shortage "
            "recovery residual."
        ),
    )
    parser.add_argument(
        "--freeze-fire-actors-for-dish",
        action="store_true",
        help=(
            "Preserve emergency, suppression, and extinguisher-cleanup "
            "residuals while training plate-shortage recovery."
        ),
    )
    parser.add_argument(
        "--freeze-dish-actor-for-assembly",
        action="store_true",
        help=(
            "Preserve the dirty-plate pickup and washing residual while "
            "training only the post-wash assembly recovery branch."
        ),
    )
    parser.add_argument(
        "--freeze-assembly-actor-for-workflow",
        action="store_true",
        help=(
            "Preserve the broad post-wash assembly residual while training "
            "only the held bun-and-lettuce plate workflow residual."
        ),
    )
    parser.add_argument("--bc-pretrain-steps", type=int, default=0)
    parser.add_argument("--bc-batch-size", type=int, default=128)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
    parser.add_argument("--bc-aux-coef", type=float, default=0.0)
    return TrainConfig(**vars(parser.parse_args(argv)))


def main(argv: Optional[Sequence[str]] = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
