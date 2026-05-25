"""Central configuration for the ViZDoom survival project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem locations used by training, evaluation, and recording."""

    root_dir: Path = ROOT_DIR
    assets_dir: Path = ROOT_DIR / "assets"
    videos_dir: Path = ROOT_DIR / "videos"
    models_dir: Path = ROOT_DIR / "models"
    logs_dir: Path = ROOT_DIR / "logs"
    checkpoints_dir: Path = ROOT_DIR / "models" / "checkpoints"

    untrained_model: Path = ROOT_DIR / "models" / "untrained.zip"
    half_trained_model: Path = ROOT_DIR / "models" / "half_trained.zip"
    fully_trained_model: Path = ROOT_DIR / "models" / "fully_trained.zip"

    untrained_video: Path = ROOT_DIR / "videos" / "untrained.mp4"
    half_trained_video: Path = ROOT_DIR / "videos" / "half_trained.mp4"
    fully_trained_video: Path = ROOT_DIR / "videos" / "fully_trained.mp4"

    evolution_gif: Path = ROOT_DIR / "assets" / "evolution.gif"
    reward_plot: Path = ROOT_DIR / "assets" / "reward_plot.png"
    episode_length_plot: Path = ROOT_DIR / "assets" / "episode_length_plot.png"
    loss_plot: Path = ROOT_DIR / "assets" / "loss_plot.png"
    monitor_file: Path = ROOT_DIR / "logs" / "training.monitor.csv"
    progress_file: Path = ROOT_DIR / "logs" / "progress.csv"


@dataclass(frozen=True)
class EnvironmentConfig:
    """ViZDoom environment and visual preprocessing settings."""

    env_id: str = "VizdoomDefendCenter-v1"
    frame_skip: int = 4
    image_size: Tuple[int, int] = (84, 84)
    frame_stack: int = 4
    seed: int = 42


@dataclass(frozen=True)
class RewardWeights:
    """Weights for the custom shaped reward function."""

    survival: float = 0.01
    damage_dealt: float = 1.0
    item_pickup: float = 0.5
    health_lost: float = 0.02
    ammo_used: float = 0.01
    death_penalty: float = 2.0


@dataclass(frozen=True)
class TrainingConfig:
    """PPO hyperparameters chosen for laptop-friendly training."""

    total_timesteps: int = 200_000
    learning_rate: float = 2.5e-4
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.20
    ent_coef: float = 0.01
    vf_coef: float = 0.50
    max_grad_norm: float = 0.50
    features_dim: int = 256
    policy_hidden_units: int = 128
    checkpoint_freq: int = 50_000
    device: str = "cpu"


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation settings used by evaluate.py."""

    episodes: int = 10
    deterministic: bool = True
    max_steps_per_episode: int = 1_200


@dataclass(frozen=True)
class VideoConfig:
    """Video and GIF rendering settings."""

    fps: int = 30
    gif_fps: int = 8
    max_steps: int = 750
    gif_frames_per_stage: int = 80


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level project configuration."""

    paths: PathsConfig = PathsConfig()
    env: EnvironmentConfig = EnvironmentConfig()
    rewards: RewardWeights = RewardWeights()
    train: TrainingConfig = TrainingConfig()
    eval: EvaluationConfig = EvaluationConfig()
    video: VideoConfig = VideoConfig()


CONFIG = ProjectConfig()
