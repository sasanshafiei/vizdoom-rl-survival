"""Evaluate a trained or random ViZDoom PPO policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG
from src.utils import ensure_directories, optional_path, positive_int, set_global_seed
from src.wrappers import make_doom_vec_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a ViZDoom PPO model.")
    parser.add_argument(
        "--model",
        type=optional_path,
        default=CONFIG.paths.fully_trained_model,
        help="Path to a Stable-Baselines3 .zip model.",
    )
    parser.add_argument(
        "--episodes",
        type=positive_int,
        default=CONFIG.eval.episodes,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CONFIG.env.seed,
        help="Random seed.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Evaluate a random policy instead of loading a model.",
    )
    return parser.parse_args()


def evaluate(
    model_path: Optional[Path],
    episodes: int,
    seed: int,
    random_policy: bool = False,
) -> None:
    """Run deterministic evaluation and print summary statistics."""

    ensure_directories(CONFIG.paths)
    set_global_seed(seed)
    env = make_doom_vec_env(
        env_config=CONFIG.env,
        reward_weights=CONFIG.rewards,
        paths=CONFIG.paths,
        seed=seed,
        training=False,
    )

    model = None
    if not random_policy:
        if model_path is None or not model_path.exists():
            raise FileNotFoundError(
                "Model not found. Train first or pass --random. "
                f"Requested path: {model_path}"
            )
        model = PPO.load(model_path, env=env, device=CONFIG.train.device)

    rewards: list[float] = []
    lengths: list[int] = []

    for _ in range(episodes):
        observation = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done and episode_length < CONFIG.eval.max_steps_per_episode:
            if random_policy:
                action = np.array([env.action_space.sample()])
            else:
                action, _ = model.predict(
                    observation,
                    deterministic=CONFIG.eval.deterministic,
                )
            observation, reward, dones, _ = env.step(action)
            episode_reward += float(reward[0])
            episode_length += 1
            done = bool(dones[0])

        rewards.append(episode_reward)
        lengths.append(episode_length)

    env.close()
    print(f"Episodes: {episodes}")
    print(f"Mean reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"Mean length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")


def main() -> None:
    args = parse_args()
    evaluate(
        model_path=args.model,
        episodes=args.episodes,
        seed=args.seed,
        random_policy=args.random,
    )


if __name__ == "__main__":
    main()
