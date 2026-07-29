#!/usr/bin/env python3
"""Build a tamper-evident, fixed-position single-agent PPO artifact.

The artifact binds one inference actor, its authoritative environment config,
the exhaustive 40-position evaluation, and every WebUI replay by SHA-256.
This prevents a selected replay or a reused checkpoint filename from being
presented as evidence for a different model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from burger_marl.actions import Action  # noqa: E402
from burger_marl.env import BurgerGridworld  # noqa: E402
from burger_marl.mappo_env import BurgerMAPPOEnv, MAX_AGENTS  # noqa: E402
from burger_marl.training import (  # noqa: E402
    LocalActor,
    _masked_logits,
    environment_config,
    load_actor_state_dict_compatible,
)
from export_policy_replay import (  # noqa: E402
    actor_state_sha256,
    checkpoint_config,
    sha256_file,
)


CONTROL_STEP_SECONDS = 0.42
SCENARIOS = (
    "standard",
    "dirty_plate_carry_ready",
    "plate_exhausted_ready",
    "fire_recovery_ready",
)
DEFAULT_REPLAYS = (
    REPO_ROOT / "burger_lab/public/ppo-policy-replay-standard.json",
    REPO_ROOT / "burger_lab/public/ppo-policy-replay-dirty.json",
    REPO_ROOT / "burger_lab/public/ppo-policy-replay-fire.json",
)
ACCEPTANCE_GATES = {
    "minimum_standard_mean_deliveries": 2.0,
    "minimum_scenario_success_rate": 0.95,
    "minimum_fire_extinguish_rate": 1.0,
    "maximum_fire_active_steps": 16,
}


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def mean(values: Iterable[float]) -> float:
    materialized = tuple(values)
    return float(sum(materialized) / len(materialized))


def load_replay_evidence(
    paths: Iterable[Path],
    checkpoint_sha256: str,
    actor_sha256: str,
) -> list[dict[str, Any]]:
    evidence = []
    seen_scenarios: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        scenario = str(payload.get("scenarioId"))
        if scenario in seen_scenarios:
            raise ValueError(f"Duplicate WebUI replay scenario: {scenario}")
        seen_scenarios.add(scenario)
        if payload.get("algorithm") != "PPO":
            raise ValueError(f"{path} is not labeled PPO")
        if payload.get("checkpointSha256") != checkpoint_sha256:
            raise ValueError(f"{path} does not match the source checkpoint")
        if payload.get("actorStateSha256") != actor_sha256:
            raise ValueError(f"{path} does not match the source actor")
        evidence.append(
            {
                "scenario": scenario,
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(path),
                "deliveries": int(payload["deliveries"]),
                "evaluation_seed": int(payload["evaluationSeed"]),
                "start_position": list(payload["startPosition"]),
            }
        )
    required = {"standard", "dirty", "fire"}
    if seen_scenarios != required:
        raise ValueError(
            f"WebUI replay scenarios must be {sorted(required)}, "
            f"got {sorted(seen_scenarios)}"
        )
    return sorted(evidence, key=lambda item: item["scenario"])


@torch.no_grad()
def evaluate_stage(
    actor: LocalActor,
    config: Any,
    device: torch.device,
    stage: str,
    positions: list[tuple[int, int]],
) -> dict[str, Any]:
    envs: list[BurgerMAPPOEnv] = []
    observations: list[np.ndarray] = []
    available_actions: list[np.ndarray] = []
    initial_deliveries: list[int] = []
    event_counts = [
        {
            "fires_started": 0,
            "fires_extinguished": 0,
            "fire_active_steps": 0,
            "washed_plates": 0,
        }
        for _ in positions
    ]
    stay_index = Action.ACTION_TO_INDEX[Action.STAY]

    for index, position in enumerate(positions):
        env = BurgerMAPPOEnv(
            mdp=BurgerGridworld(config=environment_config(config)),
            num_players=1,
            start_stage=stage,
            randomize_player_positions=True,
            random_start_candidate_positions=(position,),
        )
        local, _, available = env.reset(seed=config.seed + index)
        envs.append(env)
        observations.append(local)
        available_actions.append(available)
        initial_deliveries.append(env.state.delivered_orders)

    actor.eval()
    for _ in range(config.horizon):
        local_batch = torch.as_tensor(
            np.stack([local[0] for local in observations]),
            dtype=torch.float32,
            device=device,
        )
        mask_batch = torch.as_tensor(
            np.stack([mask[0] for mask in available_actions]),
            dtype=torch.float32,
            device=device,
        )
        agent_ids = torch.zeros(len(envs), dtype=torch.long, device=device)
        actions = (
            _masked_logits(actor(local_batch, agent_ids), mask_batch)
            .argmax(dim=-1)
            .cpu()
            .numpy()
        )
        for index, (env, action) in enumerate(zip(envs, actions)):
            joint_actions = np.full(MAX_AGENTS, stay_index, dtype=np.int64)
            joint_actions[0] = int(action)
            local, _, _, dones, infos, available = env.step(joint_actions)
            observations[index] = local
            available_actions[index] = available
            event_types = [
                str(event["type"]) for event in infos[0].get("events", ())
            ]
            event_counts[index]["fires_started"] += event_types.count(
                "fire_started"
            )
            event_counts[index]["fires_extinguished"] += event_types.count(
                "fire_extinguished"
            )
            event_counts[index]["fire_active_steps"] += event_types.count(
                "fire_active"
            )
            event_counts[index]["washed_plates"] += event_types.count(
                "plate_washed"
            )
            if env.state.timestep < config.horizon and bool(dones[0]):
                raise RuntimeError(
                    f"{stage} at {positions[index]} terminated early"
                )

    results = []
    for index, (env, position) in enumerate(zip(envs, positions)):
        if env.state.timestep != config.horizon:
            raise RuntimeError(
                f"{stage} at {position} did not reach the full horizon"
            )
        results.append(
            {
                "position": list(position),
                "deliveries": (
                    env.state.delivered_orders - initial_deliveries[index]
                ),
                **event_counts[index],
            }
        )

    deliveries = [int(item["deliveries"]) for item in results]
    successes = [value >= 1 for value in deliveries]
    episode_minutes = config.horizon * CONTROL_STEP_SECONDS / 60.0
    metrics: dict[str, Any] = {
        "positions": len(results),
        "mean_deliveries": mean(deliveries),
        "deliveries_per_minute": mean(deliveries) / episode_minutes,
        "success_rate": mean(successes),
        "failures": [
            item["position"] for item in results if item["deliveries"] < 1
        ],
        "mean_washed_plates": mean(
            int(item["washed_plates"]) for item in results
        ),
        "mean_fires_started": mean(
            int(item["fires_started"]) for item in results
        ),
        "position_results": results,
    }
    if stage == "fire_recovery_ready":
        metrics.update(
            {
                "fire_extinguish_rate": mean(
                    int(item["fires_extinguished"]) >= 1
                    for item in results
                ),
                "mean_fire_active_steps": mean(
                    int(item["fire_active_steps"]) for item in results
                ),
                "max_fire_active_steps": max(
                    int(item["fire_active_steps"]) for item in results
                ),
            }
        )
    return metrics


def write_inference_checkpoint(
    output: Path,
    checkpoint: Mapping[str, Any],
    actor_sha256: str,
    source_checkpoint_sha256: str,
) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "nexus.burger.single-agent-ppo.actor.v1",
            "actor": checkpoint["actor"],
            "train_config": checkpoint["train_config"],
            "global_step": checkpoint.get("global_step"),
            "update": checkpoint.get("update"),
            "actor_state_sha256": actor_sha256,
            "source_checkpoint_sha256": source_checkpoint_sha256,
        },
        output,
    )
    return sha256_file(output)


def build_artifact(
    checkpoint_path: Path,
    output_path: Path,
    replay_paths: Iterable[Path],
    device_name: str,
    model_output: Path | None,
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    config = checkpoint_config(checkpoint)
    device = resolve_device(device_name)
    actor = LocalActor(
        BurgerMAPPOEnv(
            mdp=BurgerGridworld(config=environment_config(config)),
            num_players=1,
        ).observation_space[0].shape,
        config.hidden_size,
        Action.NUM_ACTIONS,
    ).to(device)
    load_actor_state_dict_compatible(actor, checkpoint["actor"], strict=True)
    actor_sha256 = actor_state_sha256(checkpoint["actor"])
    checkpoint_sha256 = sha256_file(checkpoint_path)
    positions = sorted(
        BurgerGridworld(
            config=environment_config(config)
        ).layout.valid_player_positions
    )
    if len(positions) != 40:
        raise RuntimeError(
            f"Expected 40 exhaustive walkable starts, got {len(positions)}"
        )

    scenario_metrics = {
        stage: evaluate_stage(actor, config, device, stage, positions)
        for stage in SCENARIOS
    }
    aggregate_mean_deliveries = mean(
        result["deliveries"]
        for metrics in scenario_metrics.values()
        for result in metrics["position_results"]
    )
    aggregate_dpm = aggregate_mean_deliveries / (
        config.horizon * CONTROL_STEP_SECONDS / 60.0
    )
    fire_metrics = scenario_metrics["fire_recovery_ready"]
    gates = {
        "standard_mean_deliveries": (
            scenario_metrics["standard"]["mean_deliveries"]
            >= ACCEPTANCE_GATES["minimum_standard_mean_deliveries"]
        ),
        "all_scenario_success_rates": all(
            metrics["success_rate"]
            >= ACCEPTANCE_GATES["minimum_scenario_success_rate"]
            for metrics in scenario_metrics.values()
        ),
        "fire_extinguish_rate": (
            fire_metrics["fire_extinguish_rate"]
            >= ACCEPTANCE_GATES["minimum_fire_extinguish_rate"]
        ),
        "fire_recovery_latency": (
            fire_metrics["max_fire_active_steps"]
            <= ACCEPTANCE_GATES["maximum_fire_active_steps"]
        ),
    }
    replay_evidence = load_replay_evidence(
        replay_paths, checkpoint_sha256, actor_sha256
    )
    inference_model = None
    if model_output is not None:
        inference_model = {
            "path": model_output.name,
            "sha256": write_inference_checkpoint(
                model_output,
                checkpoint,
                actor_sha256,
                checkpoint_sha256,
            ),
        }

    artifact = {
        "schema": "nexus.burger.single-agent-ppo-artifact.v1",
        "algorithm": "PPO",
        "num_agents": 1,
        "checkpoint": {
            "label": checkpoint_path.stem,
            "source_path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "actor_state_sha256": actor_sha256,
            "global_step": int(checkpoint.get("global_step", 0)),
            "update": int(checkpoint.get("update", 0)),
        },
        "inference_model": inference_model,
        "environment": {
            "action_contract": config.action_contract,
            "action_mask_contract": config.action_mask_contract,
            "world_object_contract": config.world_object_contract,
            "horizon_steps": config.horizon,
            "control_step_seconds": CONTROL_STEP_SECONDS,
        },
        "evaluation": {
            "protocol": (
                "deterministic masked argmax over every walkable start "
                "position"
            ),
            "positions_per_scenario": len(positions),
            "episodes": len(positions) * len(SCENARIOS),
            "aggregate_mean_deliveries": aggregate_mean_deliveries,
            "aggregate_deliveries_per_minute": aggregate_dpm,
            "scenarios": scenario_metrics,
        },
        "acceptance": {
            "thresholds": ACCEPTANCE_GATES,
            "gates": gates,
            "accepted": all(gates.values()),
        },
        "webui_replays": replay_evidence,
    }
    if not artifact["acceptance"]["accepted"]:
        raise RuntimeError(
            "Checkpoint failed artifact acceptance: "
            + json.dumps(gates, sort_keys=True)
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--replay",
        action="append",
        type=Path,
        dest="replays",
        help="WebUI replay to bind; defaults to standard, dirty, and fire.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Evaluation device (auto, cpu, or cuda).",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=None,
        help="Optional actor-only inference checkpoint for a release bundle.",
    )
    args = parser.parse_args()
    artifact = build_artifact(
        args.checkpoint,
        args.output,
        args.replays or DEFAULT_REPLAYS,
        args.device,
        args.model_output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "accepted": artifact["acceptance"]["accepted"],
                "episodes": artifact["evaluation"]["episodes"],
                "aggregate_mean_deliveries": artifact["evaluation"][
                    "aggregate_mean_deliveries"
                ],
                "checkpoint_sha256": artifact["checkpoint"]["sha256"],
                "actor_state_sha256": artifact["checkpoint"][
                    "actor_state_sha256"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
