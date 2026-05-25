"""Watch the trained ViZDoom agent in a live game window."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG
from src.wrappers import make_doom_vec_env


def main() -> None:
    env = make_doom_vec_env(
        env_config=CONFIG.env,
        reward_weights=CONFIG.rewards,
        paths=CONFIG.paths,
        seed=CONFIG.env.seed,
        training=False,
        render_mode="human",
    )

    model = PPO.load(
        CONFIG.paths.fully_trained_model,
        env=env,
        device=CONFIG.train.device,
    )

    observation = env.reset()

    for _ in range(1500):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, dones, _ = env.step(action)

        env.render()
        time.sleep(0.03)

        if bool(dones[0]):
            observation = env.reset()

    env.close()


if __name__ == "__main__":
    main()