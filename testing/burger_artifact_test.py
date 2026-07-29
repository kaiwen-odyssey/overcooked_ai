import sys
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from burger_marl.training import TrainConfig


SCRIPT_DIR = (
    Path(__file__).resolve().parents[1] / "burger_lab" / "scripts"
)
sys.path.insert(0, str(SCRIPT_DIR))

from export_policy_replay import (  # noqa: E402
    actor_state_sha256,
    checkpoint_config,
    validate_single_agent_ppo_checkpoint,
)


class TestSingleAgentPPOArtifact(unittest.TestCase):
    def _checkpoint(self):
        return {
            "train_config": asdict(
                TrainConfig(
                    total_env_steps=1,
                    num_envs=1,
                    rollout_length=1,
                    update_epochs=1,
                    num_minibatches=1,
                )
            ),
            "actor": {"weight": torch.arange(4, dtype=torch.float32)},
        }

    def test_replay_checkpoint_must_be_truthful_single_agent_ppo(self):
        checkpoint = self._checkpoint()
        self.assertEqual(
            validate_single_agent_ppo_checkpoint(checkpoint)["algorithm"],
            "ppo",
        )
        self.assertEqual(checkpoint_config(checkpoint).num_agents, 1)

        checkpoint["train_config"]["algorithm"] = "mappo"
        checkpoint["train_config"]["num_agents"] = 2
        with self.assertRaisesRegex(ValueError, "single-agent PPO"):
            checkpoint_config(checkpoint)

    def test_actor_hash_is_stable_and_sensitive_to_tensor_bytes(self):
        left = {"weight": torch.arange(4, dtype=torch.float32)}
        right = {"weight": torch.arange(4, dtype=torch.float32)}
        changed = {"weight": torch.arange(4, dtype=torch.float32) + 1}

        self.assertEqual(
            actor_state_sha256(left), actor_state_sha256(right)
        )
        self.assertNotEqual(
            actor_state_sha256(left), actor_state_sha256(changed)
        )

    def test_checkpoint_requires_actor_state(self):
        checkpoint = self._checkpoint()
        checkpoint["actor"] = {}
        with self.assertRaisesRegex(ValueError, "actor state"):
            validate_single_agent_ppo_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
