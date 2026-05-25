"""Environment wrappers for visual preprocessing and reward shaping."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecFrameStack,
    VecMonitor,
    VecTransposeImage,
)

from src.config import EnvironmentConfig, PathsConfig, RewardWeights


class RewardShapingWrapper(gym.Wrapper):
    """Replace the sparse/default ViZDoom reward with shaped survival reward.

    The selected scenario, VizdoomDefendCenter-v1, exposes game variables in
    this order: health, ammo. Positive native rewards represent successful
    monster hits/kills, so they are used as the damage-dealt term.
    """

    def __init__(self, env: gym.Env, weights: RewardWeights) -> None:
        super().__init__(env)
        self.weights = weights
        self.last_game_variables: Optional[np.ndarray] = None

    def reset(self, **kwargs: Any) -> tuple[Any, Dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self.last_game_variables = self._extract_game_variables(observation)
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, Dict[str, Any]]:
        observation, raw_reward, terminated, truncated, info = self.env.step(
            action
        )
        current_variables = self._extract_game_variables(observation)
        components = self._reward_components(
            raw_reward=float(raw_reward),
            current_variables=current_variables,
            terminated=terminated,
            truncated=truncated,
        )
        shaped_reward = self._weighted_reward(components)

        info = dict(info)
        info["raw_reward"] = float(raw_reward)
        info["reward_components"] = components
        self.last_game_variables = current_variables

        return observation, shaped_reward, terminated, truncated, info

    @staticmethod
    def _extract_game_variables(observation: Any) -> np.ndarray:
        if isinstance(observation, dict) and "gamevariables" in observation:
            return np.asarray(observation["gamevariables"], dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    def _reward_components(
        self,
        raw_reward: float,
        current_variables: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> Dict[str, float]:
        previous = self.last_game_variables
        if previous is None:
            previous = current_variables

        previous_health = self._safe_get(previous, index=0, default=100.0)
        current_health = self._safe_get(current_variables, index=0, default=0.0)
        previous_ammo = self._safe_get(previous, index=1, default=0.0)
        current_ammo = self._safe_get(current_variables, index=1, default=0.0)

        health_delta = current_health - previous_health
        ammo_delta = current_ammo - previous_ammo
        died = terminated and current_health <= 0.0 and not truncated

        return {
            "survival_bonus": 0.0 if terminated else 1.0,
            "damage_dealt": max(raw_reward, 0.0),
            "item_pickup": max(health_delta, 0.0),
            "health_lost": max(-health_delta, 0.0),
            "ammo_used": max(-ammo_delta, 0.0),
            "death_penalty": 1.0 if died else 0.0,
        }

    @staticmethod
    def _safe_get(values: np.ndarray, index: int, default: float) -> float:
        if len(values) <= index:
            return default
        return float(values[index])

    def _weighted_reward(self, components: Dict[str, float]) -> float:
        weights = self.weights
        reward = (
            weights.survival * components["survival_bonus"]
            + weights.damage_dealt * components["damage_dealt"]
            + weights.item_pickup * components["item_pickup"]
            - weights.health_lost * components["health_lost"]
            - weights.ammo_used * components["ammo_used"]
            - weights.death_penalty * components["death_penalty"]
        )
        return float(reward)


class DoomObservationWrapper(gym.ObservationWrapper):
    """Extract, grayscale, and resize the ViZDoom screen buffer."""

    def __init__(self, env: gym.Env, width: int, height: int) -> None:
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(height, width, 1),
            dtype=np.uint8,
        )

    def observation(self, observation: Any) -> np.ndarray:
        screen = self._extract_screen(observation)
        return self._preprocess_screen(screen)

    @staticmethod
    def _extract_screen(observation: Any) -> np.ndarray:
        if isinstance(observation, dict) and "screen" in observation:
            return np.asarray(observation["screen"])
        return np.asarray(observation)

    def _preprocess_screen(self, screen: np.ndarray) -> np.ndarray:
        screen = np.asarray(screen)

        if screen.ndim == 3 and screen.shape[0] in {1, 3, 4}:
            if screen.shape[-1] not in {1, 3, 4}:
                screen = np.transpose(screen, (1, 2, 0))

        if screen.ndim == 2:
            image = Image.fromarray(screen.astype(np.uint8), mode="L")
        elif screen.ndim == 3 and screen.shape[-1] == 1:
            image = Image.fromarray(screen[..., 0].astype(np.uint8), mode="L")
        else:
            rgb = screen[..., :3].astype(np.uint8)
            image = Image.fromarray(rgb).convert("L")

        image = image.resize(
            (self.width, self.height),
            resample=Image.Resampling.BILINEAR,
        )
        frame = np.asarray(image, dtype=np.uint8)
        return np.expand_dims(frame, axis=-1)


def make_doom_env(
    env_config: EnvironmentConfig,
    reward_weights: RewardWeights,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
) -> gym.Env:
    """Create a single Gymnasium-compatible ViZDoom environment."""

    try:
        from vizdoom import gymnasium_wrapper  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ViZDoom is not installed. Run: pip install -r requirements.txt"
        ) from exc

    make_kwargs: Dict[str, Any] = {
        "frame_skip": env_config.frame_skip,
        "treat_episode_timeout_as_truncation": True,
    }
    if render_mode is not None:
        make_kwargs["render_mode"] = render_mode

    env = gym.make(env_config.env_id, **make_kwargs)
    env = RewardShapingWrapper(env, reward_weights)
    width, height = env_config.image_size
    env = DoomObservationWrapper(env, width=width, height=height)

    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    return env


def make_doom_vec_env(
    env_config: EnvironmentConfig,
    reward_weights: RewardWeights,
    paths: PathsConfig,
    seed: int,
    training: bool,
    render_mode: Optional[str] = None,
) -> VecEnv:
    """Create a vectorized ViZDoom environment with frame stacking."""

    def _make_env() -> gym.Env:
        return make_doom_env(
            env_config=env_config,
            reward_weights=reward_weights,
            seed=seed,
            render_mode=render_mode,
        )

    env_fns: list[Callable[[], gym.Env]] = [_make_env]
    vec_env: VecEnv = DummyVecEnv(env_fns)
    vec_env.seed(seed)

    monitor_file = str(paths.monitor_file) if training else None
    vec_env = VecMonitor(vec_env, filename=monitor_file)
    vec_env = VecFrameStack(
        vec_env,
        n_stack=env_config.frame_stack,
        channels_order="last",
    )
    vec_env = VecTransposeImage(vec_env)
    return vec_env


def reward_weights_as_dict(weights: RewardWeights) -> Dict[str, float]:
    """Return reward weights as a plain dictionary for logging/reporting."""

    return {key: float(value) for key, value in asdict(weights).items()}
