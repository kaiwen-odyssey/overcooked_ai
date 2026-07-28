"""PyTorch PPO/MAPPO trainer for the authoritative burger environment.

The one-agent mode is ordinary PPO. With two or more active agents the actor
parameters are shared, each actor receives only its local observation, and the
value function receives the separate global state exposed by
``BurgerMAPPOEnv``. This is centralized training with decentralized execution
(CTDE), rather than independent PPO with a relabeled command.
"""

from __future__ import annotations

import argparse
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

from burger_marl.actions import Action
from burger_marl.mappo_env import (
    BurgerMAPPOEnv,
    MAX_AGENTS,
)
from burger_marl.env import (
    BEEF_DISPENSER,
    BUN_DISPENSER,
    FLOOR,
    GRILL,
    LETTUCE_DISPENSER,
    PLATE_RACK,
    SERVE,
    BurgerConfig,
    BurgerGridworld,
    BurgerRewardConfig,
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
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 10.0
    hidden_size: int = 256
    horizon: int = 429
    cook_steps: int = 24
    burn_steps: int = 16
    plate_return_steps: int = 12
    fire_started_penalty: float = -5.0
    eval_episodes: int = 5
    eval_interval_updates: int = 20
    checkpoint_interval_updates: int = 20
    device: str = "auto"
    output_dir: str = "runs/burger"
    actor_init: Optional[str] = None
    deterministic_torch: bool = False
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
            "horizon": self.horizon,
            "cook_steps": self.cook_steps,
            "burn_steps": self.burn_steps,
            "plate_return_steps": self.plate_return_steps,
            "bc_batch_size": self.bc_batch_size,
            "bc_learning_rate": self.bc_learning_rate,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError("Training values must be positive: {}".format(invalid))
        if self.fire_started_penalty > 0:
            raise ValueError("fire_started_penalty must not be positive")
        if self.bc_pretrain_steps < 0:
            raise ValueError("bc_pretrain_steps must not be negative")
        if self.bc_aux_coef < 0:
            raise ValueError("bc_aux_coef must not be negative")
        if self.bc_aux_coef > 0 and self.bc_pretrain_steps == 0:
            raise ValueError("bc_aux_coef requires behavior-cloning pretraining")


class LocalActor(nn.Module):
    """Parameter-shared local actor used unchanged during execution."""

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
        self.apply(_orthogonal_init)
        nn.init.orthogonal_(self.policy[-1].weight, gain=0.01)

    def forward(
        self, observations: torch.Tensor, agent_ids: torch.Tensor
    ) -> torch.Tensor:
        features = self.encoder(observations)
        identity = self.agent_embedding(agent_ids)
        return self.policy(torch.cat((features, identity), dim=-1))


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
    environment_seconds: float


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
        plate_return_steps=config.plate_return_steps,
        reward=BurgerRewardConfig(
            fire_started=config.fire_started_penalty,
        ),
    )


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
    ) -> None:
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.envs = [
            BurgerMAPPOEnv(
                mdp=BurgerGridworld(config=burger_config),
                num_players=num_agents,
            )
            for _ in range(num_envs)
        ]
        resets = [env.reset(seed=index) for index, env in enumerate(self.envs)]
        self.observations = np.stack([item[0] for item in resets])
        self.shared_observations = np.stack([item[1] for item in resets])
        self.available_actions = np.stack([item[2] for item in resets])
        self.episode_returns = np.zeros(num_envs, dtype=np.float64)

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
        rewards = np.asarray(
            [item[2][0, 0] for item in transitions], dtype=np.float32
        )
        dones = np.asarray(
            [item[3][0] for item in transitions], dtype=np.bool_
        )
        infos = [item[4] for item in transitions]
        self.episode_returns += rewards

        for index, done in enumerate(dones):
            if done:
                infos[index][0]["episode_deliveries"] = int(
                    self.envs[index].state.delivered_orders
                )
                reset_obs, reset_shared, reset_available = self.envs[index].reset()
                observations[index] = reset_obs
                shared[index] = reset_shared
                self.available_actions[index] = reset_available

        self.observations = observations
        self.shared_observations = shared
        return observations, shared, rewards, dones, infos


class PPOTrainer:
    """Shared implementation: PPO for one actor, MAPPO for a team."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
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
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config.learning_rate,
            eps=1e-5,
        )
        self.vector_env = BurgerVectorEnv(
            config.num_envs,
            config.num_agents,
            environment_config(config),
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
                self.actor.parameters(), self.config.max_grad_norm
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
            checkpoint_path, map_location=self.device, weights_only=False
        )
        state_dict = checkpoint.get("actor", checkpoint)
        missing, unexpected = self.actor.load_state_dict(
            state_dict, strict=False
        )
        allowed_missing = {"agent_embedding.weight"}
        if unexpected or set(missing) - allowed_missing:
            raise ValueError(
                "Actor checkpoint mismatch: missing={}, unexpected={}".format(
                    missing, unexpected
                )
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
            )
            distribution = Categorical(logits=logits)
            active_actions = distribution.sample()
            active_log_probs = distribution.log_prob(active_actions)
            team_values = self.critic(shared)

            joint_actions = np.full(
                (config.num_envs, MAX_AGENTS),
                stay_index,
                dtype=np.int64,
            )
            joint_actions[:, : config.num_agents] = (
                active_actions.view(config.num_envs, config.num_agents)
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
                collisions += event_types.count("collision")
                fires += event_types.count("fire_started")
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
            environment_seconds=time.perf_counter() - collect_started,
        )

    def update_policy(self, rollout: Rollout) -> Dict[str, float]:
        config = self.config
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
                logits = _masked_logits(
                    self.actor(local_batch.flatten(0, 1), agent_ids),
                    action_mask,
                )
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
                loss = (
                    policy_loss
                    + config.value_coef * value_loss
                    - config.entropy_coef * entropy
                    + config.bc_aux_coef * bc_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    list(self.actor.parameters())
                    + list(self.critic.parameters()),
                    config.max_grad_norm,
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
                metrics["entropy"].append(float(entropy.detach()))
                metrics["approx_kl"].append(float(approx_kl.detach()))
                metrics["clip_fraction"].append(float(clip_fraction.detach()))
                metrics["grad_norm"].append(float(grad_norm.detach()))

        return {
            name: float(np.mean(values)) for name, values in metrics.items()
        }

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
) -> Dict[str, float]:
    """Fixed-start deterministic evaluation; sparse success is reported alone."""

    episode_count = episodes or config.eval_episodes
    total_rewards: List[float] = []
    sparse_rewards: List[float] = []
    deliveries: List[int] = []
    collisions: List[int] = []
    fires: List[int] = []
    stay_index = Action.ACTION_TO_INDEX[Action.STAY]
    actor.eval()

    for episode in range(episode_count):
        env = BurgerMAPPOEnv(
            mdp=BurgerGridworld(config=environment_config(config)),
            num_players=config.num_agents,
        )
        observations, _, available = env.reset(seed=config.seed + episode)
        episode_reward = 0.0
        episode_sparse = 0.0
        episode_collisions = 0
        episode_fires = 0
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
            episode_collisions += event_types.count("collision")
            episode_fires += event_types.count("fire_started")
            done = bool(dones[0])
        total_rewards.append(episode_reward)
        sparse_rewards.append(episode_sparse)
        deliveries.append(env.state.delivered_orders)
        collisions.append(episode_collisions)
        fires.append(episode_fires)

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
    }


def train(config: TrainConfig) -> Path:
    trainer = PPOTrainer(config)
    run_name = "{}_{}agent_seed{}".format(
        config.algorithm, config.num_agents, config.seed
    )
    output_dir = Path(config.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
                    evaluate_policy(
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
        "elapsed_seconds": time.perf_counter() - training_started,
        "global_env_steps": trainer.global_step,
        "measured_steps_per_second": trainer.global_step
        / max(time.perf_counter() - training_started, 1e-9),
        "final_evaluation": evaluate_policy(
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
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=429)
    parser.add_argument("--cook-steps", type=int, default=24)
    parser.add_argument("--burn-steps", type=int, default=16)
    parser.add_argument("--plate-return-steps", type=int, default=12)
    parser.add_argument("--fire-started-penalty", type=float, default=-5.0)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-interval-updates", type=int, default=20)
    parser.add_argument("--checkpoint-interval-updates", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="runs/burger")
    parser.add_argument("--actor-init")
    parser.add_argument("--deterministic-torch", action="store_true")
    return TrainConfig(**vars(parser.parse_args(argv)))


def main(argv: Optional[Sequence[str]] = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
