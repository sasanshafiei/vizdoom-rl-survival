"""Train a PPO agent for the ViZDoom visual survival project."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from src.config import CONFIG, ProjectConfig
from src.model import create_ppo_model
from src.utils import ensure_directories, plot_training_curves, positive_int
from src.utils import set_global_seed
from src.wrappers import make_doom_vec_env, reward_weights_as_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO on VizdoomDefendCenter-v1."
    )
    parser.add_argument(
        "--timesteps",
        type=positive_int,
        default=CONFIG.train.total_timesteps,
        help="Total PPO training timesteps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CONFIG.env.seed,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def make_runtime_config(args: argparse.Namespace) -> ProjectConfig:
    train_config = replace(CONFIG.train, total_timesteps=args.timesteps)
    env_config = replace(CONFIG.env, seed=args.seed)
    return replace(CONFIG, train=train_config, env=env_config)


def train(config: ProjectConfig) -> None:
    """Train in two phases and save untrained, half, and full models."""

    ensure_directories(config.paths)
    set_global_seed(config.env.seed)

    env = make_doom_vec_env(
        env_config=config.env,
        reward_weights=config.rewards,
        paths=config.paths,
        seed=config.env.seed,
        training=True,
    )

    model = create_ppo_model(
        env=env,
        train_config=config.train,
        seed=config.env.seed,
    )
    logger = configure(str(config.paths.logs_dir), ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    print("Reward weights:", reward_weights_as_dict(config.rewards))
    model.save(config.paths.untrained_model)
    print(f"Saved untrained checkpoint to {config.paths.untrained_model}")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, config.train.checkpoint_freq),
        save_path=str(config.paths.checkpoints_dir),
        name_prefix="ppo_vizdoom_survival",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    half_timesteps = max(1, config.train.total_timesteps // 2)
    remaining_timesteps = config.train.total_timesteps - half_timesteps

    model.learn(
        total_timesteps=half_timesteps,
        callback=checkpoint_callback,
        reset_num_timesteps=True,
        progress_bar=True,
    )
    model.save(config.paths.half_trained_model)
    print(f"Saved half-trained checkpoint to {config.paths.half_trained_model}")

    if remaining_timesteps > 0:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=False,
            progress_bar=True,
        )

    model.save(config.paths.fully_trained_model)
    print(f"Saved fully trained checkpoint to {config.paths.fully_trained_model}")
    env.close()

    try:
        plot_training_curves(config.paths)
        print("Saved training plots to assets/.")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Training finished, but plots were not generated: {exc}")


def main() -> None:
    args = parse_args()
    config = make_runtime_config(args)
    train(config)


if __name__ == "__main__":
    main()
