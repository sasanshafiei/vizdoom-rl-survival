"""Record videos and an evolution GIF for the three training stages."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG
from src.utils import ensure_directories, set_global_seed
from src.wrappers import make_doom_vec_env


@dataclass(frozen=True)
class Stage:
    """A video stage in the training progression."""

    key: str
    label: str
    model_path: Optional[Path]
    video_path: Path
    random_policy: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record ViZDoom stage videos.")
    parser.add_argument(
        "--stage",
        choices=["all", "untrained", "half", "full"],
        default="all",
        help="Which stage to record.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CONFIG.env.seed,
        help="Random seed.",
    )
    return parser.parse_args()


def build_stages() -> Dict[str, Stage]:
    """Define untrained, half-trained, and fully trained recording stages."""

    return {
        "untrained": Stage(
            key="untrained",
            label="Stage 1: Untrained / Random Agent",
            model_path=CONFIG.paths.untrained_model,
            video_path=CONFIG.paths.untrained_video,
            random_policy=True,
        ),
        "half": Stage(
            key="half",
            label="Stage 2: Half-Trained Agent",
            model_path=CONFIG.paths.half_trained_model,
            video_path=CONFIG.paths.half_trained_video,
        ),
        "full": Stage(
            key="full",
            label="Stage 3: Fully Trained Agent",
            model_path=CONFIG.paths.fully_trained_model,
            video_path=CONFIG.paths.fully_trained_video,
        ),
    }


def record_stage(stage: Stage, seed: int) -> List[np.ndarray]:
    """Record one video stage and return sampled frames for the GIF."""

    env = make_doom_vec_env(
        env_config=CONFIG.env,
        reward_weights=CONFIG.rewards,
        paths=CONFIG.paths,
        seed=seed,
        training=False,
        render_mode="rgb_array",
    )

    model = None
    if not stage.random_policy:
        if stage.model_path is None or not stage.model_path.exists():
            env.close()
            raise FileNotFoundError(
                f"Missing model for {stage.label}: {stage.model_path}. "
                "Run training before recording this stage."
            )
        model = PPO.load(stage.model_path, env=env, device=CONFIG.train.device)

    observation = env.reset()
    done = False
    frames: List[np.ndarray] = []

    with imageio.get_writer(stage.video_path, fps=CONFIG.video.fps) as writer:
        for step in range(CONFIG.video.max_steps):
            frame = capture_frame(env)
            frame = add_label(frame, stage.label, step)
            writer.append_data(frame)
            frames.append(frame)

            if stage.random_policy:
                action = np.array([env.action_space.sample()])
            else:
                action, _ = model.predict(observation, deterministic=True)

            observation, _, dones, _ = env.step(action)
            done = bool(dones[0])
            if done:
                break

    env.close()
    print(f"Saved {stage.label} video to {stage.video_path}")
    return sample_frames(frames, CONFIG.video.gif_frames_per_stage)


def capture_frame(env: object) -> np.ndarray:
    """Render a frame from a vectorized environment."""

    frame = env.render()
    if isinstance(frame, list):
        frame = frame[0]
    if frame is None:
        raise RuntimeError("Environment returned no frame. Check render_mode.")
    frame_array = np.asarray(frame)
    if frame_array.ndim == 4:
        frame_array = frame_array[0]
    return frame_array.astype(np.uint8)


def add_label(frame: np.ndarray, label: str, step: int) -> np.ndarray:
    """Overlay a readable stage label on a video frame."""

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text = f"{label} | step {step}"
    padding = 6
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    width = right - left
    height = bottom - top
    draw.rectangle(
        [0, 0, width + 2 * padding, height + 2 * padding],
        fill=(0, 0, 0),
    )
    draw.text((padding, padding), text, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def sample_frames(frames: List[np.ndarray], max_frames: int) -> List[np.ndarray]:
    """Downsample frames so the README GIF stays small."""

    if len(frames) <= max_frames:
        return frames
    indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
    return [frames[index] for index in indices]


def save_evolution_gif(frames: List[np.ndarray]) -> None:
    """Save the sequential three-stage evolution GIF."""

    if not frames:
        raise ValueError("No frames available for GIF generation.")
    imageio.mimsave(
        CONFIG.paths.evolution_gif,
        frames,
        fps=CONFIG.video.gif_fps,
    )
    print(f"Saved evolution GIF to {CONFIG.paths.evolution_gif}")


def main() -> None:
    args = parse_args()
    ensure_directories(CONFIG.paths)
    set_global_seed(args.seed)

    stages = build_stages()
    if args.stage == "all":
        selected_keys = ["untrained", "half", "full"]
    else:
        selected_keys = [args.stage]

    gif_frames: List[np.ndarray] = []
    for key in selected_keys:
        gif_frames.extend(record_stage(stages[key], args.seed))

    if args.stage == "all":
        save_evolution_gif(gif_frames)


if __name__ == "__main__":
    main()
