#!/usr/bin/env python3
"""Export a full deterministic PPO evaluation episode for the WebUI.

The JSON frames are produced by the same authoritative environment, action
masks, and actor network used by evaluation.  The WebUI only renders these
frames; it does not reimplement policy inference or mutate environment state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from burger_marl.actions import Action  # noqa: E402
from burger_marl.env import (  # noqa: E402
    CURRICULUM_START_STAGES,
    BurgerGridworld,
)
from burger_marl.mappo_env import BurgerMAPPOEnv, MAX_AGENTS  # noqa: E402
from burger_marl.training import (  # noqa: E402
    LocalActor,
    TrainConfig,
    _masked_logits,
    environment_config,
    load_actor_state_dict_compatible,
)


COUNTERS = (
    (0, 3),
    (1, 3),
    (3, 3),
    (4, 3),
    (5, 3),
    (7, 3),
    (3, 4),
    (4, 4),
    (2, 0),
    (4, 0),
    (2, 7),
    (3, 7),
    (4, 7),
    (5, 7),
)
COUNTER_INDEX = {point: index for index, point in enumerate(COUNTERS)}
ACTION_LABELS = ("↑", "↓", "→", "←", "STAY", "PICK/DROP", "PROCESS")
ORIENTATION_LABELS = {
    (0, -1): "north",
    (0, 1): "south",
    (1, 0): "east",
    (-1, 0): "west",
}
EVENT_LABELS = {
    "clean_plate_pickup": "PICK/DROP · PICK CLEAN PLATE",
    "clean_plate_return": "PICK/DROP · RETURN CLEAN PLATE",
    "dispenser_pickup": "PICK/DROP · PICK INGREDIENT",
    "ingredient_added_from_dispenser": "PICK/DROP · ADD INGREDIENT",
    "ingredient_added_to_held_plate": "PICK/DROP · ADD FROM COUNTER",
    "ingredient_added_to_counter_plate": "PICK/DROP · PLATE ON COUNTER",
    "counter_pickup": "PICK/DROP · PICK FROM COUNTER",
    "counter_drop": "PICK/DROP · DROP ON COUNTER",
    "raw_beef_placed": "PICK/DROP · PLACE RAW BEEF",
    "cooked_beef_added_to_held_plate": "PICK/DROP · PLATE COOKED BEEF",
    "correct_delivery": "PICK/DROP · SERVE BURGER",
    "dirty_plate_pickup": "PICK/DROP · PICK DIRTY PLATE",
    "wash_started": "PICK/DROP · LOAD SINK",
    "wash_progress": "PROCESS · WASH PLATE",
    "plate_washed": "PROCESS · PLATE CLEAN",
    "extinguisher_pickup": "PICK/DROP · PICK EXTINGUISHER",
    "extinguisher_return": "PICK/DROP · RETURN EXTINGUISHER",
    "fire_extinguished": "PROCESS · EXTINGUISH FIRE",
    "food_discarded": "PICK/DROP · DISCARD FOOD",
    "plate_contents_discarded": "PICK/DROP · EMPTY PLATE",
    "beef_ready": "BEEF READY",
    "fire_started": "FIRE STARTED",
    "dirty_plates_returned": "DIRTY PLATE RETURNED",
}
TASK_PROGRESS_EVENTS = {
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
MAX_STAGNATION_STEPS = 60
SCENARIO_METADATA = {
    "standard": {
        "id": "standard",
        "label": "标准生产",
        "curriculum": "random_standard_180s",
        "required_event": "plate_washed",
    },
    "fire_recovery_ready": {
        "id": "fire",
        "label": "灭火恢复",
        "curriculum": "fire_recovery_ready_180s",
        "required_event": "fire_extinguished",
    },
    "dirty_plate_carry_ready": {
        "id": "dirty",
        "label": "脏盘回收",
        "curriculum": "dirty_plate_carry_ready_180s",
        "required_event": "plate_washed",
    },
}


def ui_item(item: str | None) -> str | None:
    return item.replace("_", "-") if item is not None else None


def validate_single_agent_ppo_checkpoint(
    checkpoint: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Reject checkpoints that cannot truthfully be labeled one-agent PPO."""

    raw = checkpoint.get("train_config")
    if not isinstance(raw, dict):
        raise ValueError("Checkpoint is missing train_config")
    if raw.get("algorithm") != "ppo" or raw.get("num_agents") != 1:
        raise ValueError(
            "WebUI replay requires a single-agent PPO checkpoint "
            "(train_config.algorithm='ppo', train_config.num_agents=1)"
        )
    actor = checkpoint.get("actor")
    if not isinstance(actor, dict) or not actor:
        raise ValueError("Checkpoint is missing a non-empty actor state dict")
    return raw


def checkpoint_config(checkpoint: dict[str, Any]) -> TrainConfig:
    raw = validate_single_agent_ppo_checkpoint(checkpoint)
    allowed = {field.name for field in fields(TrainConfig)}
    values = {key: value for key, value in raw.items() if key in allowed}
    values.update(
        {
            "device": "cpu",
            "num_envs": 1,
            "num_agents": 1,
            "algorithm": "ppo",
            "actor_init": None,
            "actor_anchor_kl_coef": 0.0,
            "training_start_stage": "standard",
            "training_start_mix_stage": None,
            "randomize_start_positions": True,
            "random_start_max_objective_distance": None,
            "random_start_candidate_positions": (),
            "training_episode_steps": None,
            "freeze_base_actor_for_emergency": False,
            "freeze_cleanup_actor_for_fire": False,
            "freeze_emergency_actor_for_suppression": False,
            "freeze_suppression_actor_for_emergency": False,
            "freeze_base_actor_for_dish": False,
            "freeze_fire_actors_for_dish": False,
            "freeze_dish_actor_for_assembly": False,
            "freeze_assembly_actor_for_workflow": False,
            "raw_beef_placed_reward": 0.0,
            "dirty_plate_pickup_reward": 0.0,
            "wash_started_reward": 0.0,
            "wash_progress_reward": 0.0,
            "plate_washed_reward": 0.0,
            "fire_extinguished_reward": 0.0,
            "allow_mixed_curriculum_event_rewards": False,
            "bc_pretrain_steps": 0,
        }
    )
    return TrainConfig(**values)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def actor_state_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, shapes, dtypes, and bytes independent of torch.save."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Actor state entry {name!r} is not a tensor")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def motion_state(
    env: BurgerMAPPOEnv,
    action_label: str,
    interaction_label: str | None,
    fires_started: int,
    fires_extinguished: int,
    washed_plates: int,
    discarded_items: int,
    initial_deliveries: int,
) -> dict[str, Any]:
    state = env.state
    player = state.players[0]
    counter_objects = {
        str(COUNTER_INDEX[position]): ui_item(item)
        for position, item in state.counter_objects.items()
        if position in COUNTER_INDEX
    }
    grill_food = ui_item(state.grill.food)
    work_active = (
        state.sink.has_dirty_plate
        and action_label == "PROCESS"
        and interaction_label is not None
    )
    return {
        "agents": [
            {
                "position": list(player.position),
                "cursor": state.timestep,
                "orientation": ORIENTATION_LABELS[player.orientation],
                "lastAction": action_label,
                "carrying": ui_item(player.held_object),
                "interactionCount": state.timestep,
                "interactionLabel": interaction_label,
                "workKind": "washing" if work_active else None,
                "workTicksRemaining": (
                    max(0, env.mdp.config.wash_steps - state.sink.wash_progress)
                    if work_active
                    else 0
                ),
                "workTotalTicks": env.mdp.config.wash_steps if work_active else 0,
                "pendingCarry": None,
                "blocked": False,
                "blockedReason": None,
                "blockedKind": None,
            }
        ],
        "step": state.timestep,
        "avoidedCollisions": 0,
        "dirtyPlatesAtReturn": state.dirty_plates_at_return,
        "dirtyPlatesInSink": int(state.sink.has_dirty_plate),
        "pendingPlateReturns": list(state.pending_plate_returns),
        "cleanPlatesAtRack": state.clean_plates,
        "washedPlates": washed_plates,
        "counterObjects": counter_objects,
        "discardedItems": discarded_items,
        "firesStarted": fires_started,
        "firesExtinguished": fires_extinguished,
        "extinguisherAtStation": state.extinguisher_available,
        "grillFood": grill_food,
        "grillCookTicksRemaining": (
            max(0, env.mdp.config.cook_steps - state.grill.cook_ticks)
            if state.grill.food == "raw_beef"
            else 0
        ),
        "grillBurnTicksRemaining": (
            max(0, env.mdp.config.burn_steps - state.grill.ready_ticks)
            if state.grill.food == "cooked_beef"
            else 0
        ),
        "servedOrders": state.delivered_orders - initial_deliveries,
    }


def export_replay(
    checkpoint_path: Path,
    output_path: Path,
    evaluation_seed: int | None = None,
    start_stage: str = "standard",
    randomize_start_positions: bool = False,
) -> None:
    if start_stage not in SCENARIO_METADATA:
        raise ValueError(
            "WebUI replay exports support only "
            f"{', '.join(SCENARIO_METADATA)}"
        )
    scenario = SCENARIO_METADATA[start_stage]
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    config = checkpoint_config(checkpoint)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    actor_sha256 = actor_state_sha256(checkpoint["actor"])
    env = BurgerMAPPOEnv(
        mdp=BurgerGridworld(config=environment_config(config)),
        num_players=1,
        start_stage=start_stage,
        randomize_player_positions=randomize_start_positions,
    )
    seed = config.seed if evaluation_seed is None else evaluation_seed
    observations, _, available = env.reset(seed=seed)
    initial_deliveries = env.state.delivered_orders
    start_position = list(env.state.players[0].position)
    actor = LocalActor(
        env.observation_space[0].shape,
        config.hidden_size,
        Action.NUM_ACTIONS,
    )
    load_actor_state_dict_compatible(
        actor, checkpoint["actor"], strict=True
    )
    actor.eval()

    cumulative_reward = 0.0
    cumulative_sparse_reward = 0.0
    fires_started = 0
    fires_extinguished = 0
    washed_plates = 0
    discarded_items = 0
    delivered_at: int | None = None
    frames: list[dict[str, Any]] = [
        {
            "state": motion_state(
                env,
                "STAY",
                None,
                0,
                0,
                0,
                0,
                initial_deliveries,
            ),
            "actionIndex": Action.ACTION_TO_INDEX[Action.STAY],
            "action": "STAY",
            "reward": 0.0,
            "cumulativeReward": 0.0,
            "cumulativeSparseReward": 0.0,
            "events": [],
        }
    ]

    with torch.no_grad():
        while env.state.timestep < config.horizon:
            local = torch.as_tensor(
                observations[:1], dtype=torch.float32
            )
            mask = torch.as_tensor(available[:1], dtype=torch.float32)
            logits = _masked_logits(
                actor(local, torch.zeros(1, dtype=torch.long)),
                mask,
            )
            action_index = int(logits.argmax(dim=-1).item())
            joint_actions = np.full(
                MAX_AGENTS,
                Action.ACTION_TO_INDEX[Action.STAY],
                dtype=np.int64,
            )
            joint_actions[0] = action_index
            (
                observations,
                _,
                rewards,
                dones,
                infos,
                available,
            ) = env.step(joint_actions)
            reward = float(rewards[0, 0])
            cumulative_reward += reward
            breakdown = infos[0]["reward_breakdown"]
            cumulative_sparse_reward += float(breakdown["correct_delivery"])
            events = [dict(event) for event in infos[0]["events"]]
            event_types = [str(event["type"]) for event in events]
            fires_started += event_types.count("fire_started")
            fires_extinguished += event_types.count("fire_extinguished")
            washed_plates += event_types.count("plate_washed")
            discarded_items += sum(
                event_type in {"food_discarded", "plate_contents_discarded"}
                for event_type in event_types
            )
            interaction_label = next(
                (
                    EVENT_LABELS[event_type]
                    for event_type in event_types
                    if event_type in EVENT_LABELS
                ),
                None,
            )
            action_label = ACTION_LABELS[action_index]
            frames.append(
                {
                    "state": motion_state(
                        env,
                        action_label,
                        interaction_label,
                        fires_started,
                        fires_extinguished,
                        washed_plates,
                        discarded_items,
                        initial_deliveries,
                    ),
                    "actionIndex": action_index,
                    "action": action_label,
                    "reward": reward,
                    "cumulativeReward": cumulative_reward,
                    "cumulativeSparseReward": cumulative_sparse_reward,
                    "events": event_types,
                }
            )
            if (
                env.state.delivered_orders > initial_deliveries
                and delivered_at is None
            ):
                delivered_at = env.state.timestep
            if bool(dones[0]):
                break

    if env.state.timestep != config.horizon:
        raise RuntimeError(
            "Refusing to export a truncated replay: expected "
            f"{config.horizon} steps, got {env.state.timestep}"
        )
    deliveries = env.state.delivered_orders - initial_deliveries
    if deliveries < 1:
        raise RuntimeError(
            "Refusing to export an unvalidated replay: expected a delivery, "
            f"got {deliveries}"
        )
    all_event_types = {
        event_type
        for frame in frames
        for event_type in frame["events"]
    }
    required_event = str(scenario["required_event"])
    if required_event not in all_event_types:
        raise RuntimeError(
            "Refusing to export a replay without required scenario event "
            f"{required_event!r}"
        )
    if fires_started != 0:
        raise RuntimeError(
            "Refusing to export a replay that starts a fire"
        )
    stagnation_steps = 0
    max_stagnation_steps = 0
    for previous, current in zip(frames, frames[1:]):
        previous_position = previous["state"]["agents"][0]["position"]
        current_position = current["state"]["agents"][0]["position"]
        task_progress = bool(
            TASK_PROGRESS_EVENTS.intersection(current["events"])
        )
        if current_position != previous_position or task_progress:
            stagnation_steps = 0
        else:
            stagnation_steps += 1
            max_stagnation_steps = max(
                max_stagnation_steps, stagnation_steps
            )
    if max_stagnation_steps > MAX_STAGNATION_STEPS:
        raise RuntimeError(
            "Refusing to export a visibly stalled replay: longest "
            f"nonproductive stationary run is {max_stagnation_steps} steps"
        )

    payload = {
        "schema": "nexus.burger.ppo-policy-replay.v1",
        "algorithm": "PPO",
        "execution": "deterministic argmax with authoritative action masks",
        "checkpoint": str(checkpoint_path),
        "checkpointLabel": checkpoint_path.stem,
        "checkpointSha256": checkpoint_sha256,
        "actorStateSha256": actor_sha256,
        "scenarioId": scenario["id"],
        "scenarioLabel": scenario["label"],
        "startStage": start_stage,
        "randomizedStart": randomize_start_positions,
        "curriculumStage": scenario["curriculum"],
        "environment": "standalone_v2_adjacent_pick_drop_process",
        "controlStepSeconds": 0.42,
        "evaluationSeed": seed,
        "startPosition": start_position,
        "validatedEpisodes": 1,
        "validatedSuccesses": 1,
        "deliveries": deliveries,
        "washedPlates": washed_plates,
        "fires": fires_started,
        "firesExtinguished": fires_extinguished,
        "maxStagnationSteps": max_stagnation_steps,
        "maxStagnationSeconds": max_stagnation_steps * 0.42,
        "deliveryStep": delivered_at,
        "totalReward": cumulative_reward,
        "sparseReward": cumulative_sparse_reward,
        "frames": frames,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "frames": len(frames),
                "episode_seconds": config.horizon * 0.42,
                "evaluation_seed": seed,
                "start_position": start_position,
                "scenario": scenario["id"],
                "start_stage": start_stage,
                "randomized_start": randomize_start_positions,
                "deliveries": deliveries,
                "washed_plates": washed_plates,
                "max_stagnation_steps": max_stagnation_steps,
                "delivery_step": delivered_at,
                "total_reward": cumulative_reward,
                "fires": fires_started,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Evaluation seed; defaults to the checkpoint training seed.",
    )
    parser.add_argument(
        "--start-stage",
        choices=tuple(
            stage
            for stage in CURRICULUM_START_STAGES
            if stage in SCENARIO_METADATA
        ),
        default="standard",
        help="Authoritative environment reset stage to render.",
    )
    parser.add_argument(
        "--randomize-start-positions",
        action="store_true",
        help="Randomize the player position within the selected stage.",
    )
    args = parser.parse_args()
    export_replay(
        args.checkpoint,
        args.output,
        evaluation_seed=args.seed,
        start_stage=args.start_stage,
        randomize_start_positions=args.randomize_start_positions,
    )


if __name__ == "__main__":
    main()
