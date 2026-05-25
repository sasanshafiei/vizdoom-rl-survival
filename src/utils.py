"""Utility functions for reproducibility, plotting, and project hygiene."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import CONFIG, PathsConfig


def set_global_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_directories(paths: PathsConfig) -> None:
    """Create project output directories if they do not exist."""

    for directory in (
        paths.assets_dir,
        paths.videos_dir,
        paths.models_dir,
        paths.logs_dir,
        paths.checkpoints_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def moving_average(values: Iterable[float], window: int = 20) -> np.ndarray:
    """Compute a simple moving average for smoother training plots."""

    series = pd.Series(list(values), dtype=float)
    if series.empty:
        return np.array([], dtype=float)
    actual_window = max(1, min(window, len(series)))
    return series.rolling(actual_window, min_periods=1).mean().to_numpy()


def read_monitor_data(logs_dir: Path) -> pd.DataFrame:
    """Read Stable-Baselines3 monitor CSV files from the logs directory."""

    monitor_files = sorted(logs_dir.glob("*monitor.csv"))
    if not monitor_files:
        raise FileNotFoundError(
            f"No monitor CSV files found in {logs_dir}. Run training first."
        )

    frames = []
    for file_path in monitor_files:
        frame = pd.read_csv(file_path, comment="#")
        if {"r", "l"}.issubset(frame.columns):
            frames.append(frame)

    if not frames:
        raise ValueError("Monitor files were found, but no episode data exists.")

    data = pd.concat(frames, ignore_index=True)
    data["episode"] = np.arange(1, len(data) + 1)
    return data


def plot_training_curves(paths: PathsConfig = CONFIG.paths) -> None:
    """Create reward, episode length, and loss plots for the README."""

    ensure_directories(paths)
    monitor_data = read_monitor_data(paths.logs_dir)
    _plot_metric(
        x=monitor_data["episode"],
        y=monitor_data["r"],
        y_smooth=moving_average(monitor_data["r"]),
        title="Episode Reward During PPO Training",
        xlabel="Episode",
        ylabel="Shaped episode reward",
        output_path=paths.reward_plot,
    )
    _plot_metric(
        x=monitor_data["episode"],
        y=monitor_data["l"],
        y_smooth=moving_average(monitor_data["l"]),
        title="Episode Length During PPO Training",
        xlabel="Episode",
        ylabel="Episode length (environment steps)",
        output_path=paths.episode_length_plot,
    )
    plot_loss_curve(paths.progress_file, paths.loss_plot)


def plot_loss_curve(progress_file: Path, output_path: Path) -> None:
    """Plot PPO training loss if progress.csv is available."""

    if not progress_file.exists():
        return

    progress = pd.read_csv(progress_file)
    loss_columns = [column for column in progress.columns if "loss" in column]
    if not loss_columns:
        return

    x_column = "time/total_timesteps"
    x = progress[x_column] if x_column in progress.columns else progress.index

    plt.figure(figsize=(8, 4.5))
    for column in loss_columns:
        values = progress[column].dropna()
        if values.empty:
            continue
        plt.plot(x.loc[values.index] if hasattr(x, "loc") else x, values, label=column)
    plt.title("PPO Training Loss")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _plot_metric(
    x: pd.Series,
    y: pd.Series,
    y_smooth: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y, alpha=0.35, label="Episode value")
    plt.plot(x, y_smooth, linewidth=2, label="Moving average")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def positive_int(value: str) -> int:
    """Argparse helper that accepts only positive integers."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def optional_path(value: Optional[str]) -> Optional[Path]:
    """Convert a string path to Path while preserving None."""

    return Path(value) if value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Project utility commands.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Regenerate README plots from logs/*.monitor.csv.",
    )
    args = parser.parse_args()

    if args.plot:
        plot_training_curves(CONFIG.paths)
        print("Saved reward, episode length, and loss plots to assets/.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
