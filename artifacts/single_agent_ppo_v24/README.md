# Single-agent PPO v24 artifact

This artifact is the reproducible handoff for the single-agent burger PPO
policy deployed in the WebUI. It is an inference artifact, not a resumable
training checkpoint: optimizer and critic state are intentionally excluded.

## Identity

- Algorithm: single-agent PPO
- Checkpoint label: `checkpoint_000004`
- Source checkpoint SHA-256:
  `9277c8990f6465dc46c66cdd4f72679139433ccd771e80e471a3e964e5cd5bbe`
- Actor-state SHA-256:
  `5639323da8cb1710476674c571ef60fd682b64b2d0f50d9757584c5447ef2cb3`
- Action contract: `standalone_v2_adjacent_pick_drop_process`
- Action-mask contract: `local_visible_adjacent_context_v2`

The machine-readable evaluation and replay checksums live in the
[`ppo-single-agent-artifact.json`](../../burger_lab/public/ppo-single-agent-artifact.json)
manifest.

## Fixed-position evaluation

Deterministic masked-argmax inference was evaluated from every one of the 40
walkable start positions in each scenario, for 160 full 429-step episodes.

| Scenario | Mean deliveries / episode | Success |
| --- | ---: | ---: |
| Standard | 5.675 | 40/40 |
| Dirty-plate recovery | 5.225 | 39/40 |
| Plate-exhausted recovery | 4.700 | 39/40 |
| Fire recovery | 5.250 | 40/40 |
| Joint mean | 5.2125 | — |

Fire recovery extinguished 40/40 fires. The worst fire remained active for 14
steps. The artifact acceptance gate requires at least 95% delivery success in
every scenario, 100% fire extinguishing, and no more than 16 active-fire steps.

## Rebuild

From the repository root:

```bash
CHECKPOINT_PATH="runs/burger_single_ppo_v24/workflow_residual_dirty_65k/"\
"ppo_1agent_v2_seed20261018/checkpoint_000004.pt"
PYTHONPATH=src .venv/bin/python \
  burger_lab/scripts/build_single_agent_ppo_artifact.py \
  "$CHECKPOINT_PATH" \
  burger_lab/public/ppo-single-agent-artifact.json \
  --model-output /tmp/single-agent-ppo-v24/single-agent-ppo-actor.pt
```

The builder fails closed if a replay comes from another checkpoint or actor,
if the checkpoint is not truthful single-agent PPO, or if an acceptance gate
fails.
