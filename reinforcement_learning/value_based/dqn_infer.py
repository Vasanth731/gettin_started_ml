"""
infer.py  –  Load saved DQN weights and watch Pacman play
==========================================================
Run:
    python infer.py
"""
import ale_py
import gymnasium as gym
gym.register_envs(ale_py)
import collections
import numpy as np
import cv2
import torch
import torch.nn as nn
import gymnasium as gym

# ── Must match train.py exactly ──────────────────────────────────────────────
ENV_ID      = "ALE/Pacman-v5"
SAVE_PATH   = "pacman_best.pth"
FRAME_H     = 84
FRAME_W     = 84
FRAME_STACK = 4


# ── Frame preprocessing (same as training) ────────────────────────────────────
def preprocess(frame: np.ndarray) -> np.ndarray:
    gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
    return resized


class FrameStack:
    def __init__(self, n: int = FRAME_STACK):
        self.n = n
        self.frames = collections.deque(maxlen=n)

    def reset(self, frame):
        f = preprocess(frame)
        for _ in range(self.n):
            self.frames.append(f)
        return self._get()

    def step(self, frame):
        self.frames.append(preprocess(frame))
        return self._get()

    def _get(self) -> np.ndarray:
        return np.stack(self.frames, axis=0).astype(np.float32) / 255.0


# ── CNN Policy (same architecture as training) ────────────────────────────────
class DQN(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


# ── Action labels for display ─────────────────────────────────────────────────
ACTION_NAMES = {0: "NOOP", 1: "UP", 2: "RIGHT", 3: "LEFT", 4: "DOWN"}


# ── Inference ─────────────────────────────────────────────────────────────────
def infer(n_episodes: int = 5, render: bool = True, epsilon: float = 0.05):
    """
    Watch the trained agent play Pacman.

    Args:
        n_episodes : number of games to play
        render     : show the game window
        epsilon    : small random action probability to avoid getting stuck
                     (0.0 = fully greedy, 0.05 = 5% random)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    checkpoint = torch.load(SAVE_PATH, weights_only=False, map_location=device)
    n_actions  = checkpoint.get("n_actions", 5)

    model = DQN(n_actions).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded weights  →  trained ep={checkpoint['episode']}  "
          f"mean-100={checkpoint['best_mean_reward']:.1f}")
    print(f"Playing {n_episodes} episodes  (epsilon={epsilon})\n")

    render_mode = "human" if render else None
    env     = gym.make(ENV_ID, render_mode=render_mode)
    stacker = FrameStack()

    total_rewards = []

    for ep in range(1, n_episodes + 1):
        obs, _  = env.reset()
        state   = stacker.reset(obs)
        total   = 0.0
        steps   = 0
        actions_taken = collections.Counter()

        while True:
            # Mostly greedy, tiny epsilon to break loops
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    q_vals  = model(state_t)
                    action  = q_vals.argmax(1).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            state  = stacker.step(obs)
            total += reward
            steps += 1
            actions_taken[ACTION_NAMES[action]] += 1

            if terminated or truncated:
                break

        total_rewards.append(total)
        print(f"  Ep {ep:2d}: score = {total:8.1f}   steps = {steps:5d}   "
              f"actions = { dict(actions_taken) }")

    print(f"\n{'─'*60}")
    print(f"  Mean score : {np.mean(total_rewards):.1f}")
    print(f"  Best score : {max(total_rewards):.1f}")
    print(f"  Worst score: {min(total_rewards):.1f}")
    env.close()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    infer(
        n_episodes = 5,
        render     = True,
        epsilon    = 0.05,    # small randomness to avoid getting stuck in loops
    )