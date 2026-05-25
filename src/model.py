"""PPO model and compact CNN feature extractor."""

from __future__ import annotations

from typing import Any, Dict

import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnv
from torch import nn

from src.config import TrainingConfig


class SmallDoomCNN(BaseFeaturesExtractor):
    """Small CNN designed for 84x84 stacked grayscale ViZDoom frames."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.as_tensor(
                observation_space.sample()[None],
                dtype=torch.float32,
            )
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def build_policy_kwargs(train_config: TrainingConfig) -> Dict[str, Any]:
    """Build Stable-Baselines3 policy kwargs for the custom CNN."""

    return {
        "features_extractor_class": SmallDoomCNN,
        "features_extractor_kwargs": {
            "features_dim": train_config.features_dim,
        },
        "net_arch": {
            "pi": [train_config.policy_hidden_units],
            "vf": [train_config.policy_hidden_units],
        },
        "activation_fn": nn.ReLU,
        "normalize_images": True,
    }


def create_ppo_model(
    env: VecEnv,
    train_config: TrainingConfig,
    seed: int,
) -> PPO:
    """Create a PPO model with a CNN policy."""

    return PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=train_config.learning_rate,
        n_steps=train_config.n_steps,
        batch_size=train_config.batch_size,
        n_epochs=train_config.n_epochs,
        gamma=train_config.gamma,
        gae_lambda=train_config.gae_lambda,
        clip_range=train_config.clip_range,
        ent_coef=train_config.ent_coef,
        vf_coef=train_config.vf_coef,
        max_grad_norm=train_config.max_grad_norm,
        policy_kwargs=build_policy_kwargs(train_config),
        seed=seed,
        device=train_config.device,
        verbose=1,
    )
