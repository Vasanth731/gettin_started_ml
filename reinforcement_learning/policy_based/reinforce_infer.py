import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# ── Must match the architecture used during training ─────────────────────────
HIDDEN_SIZE = 128
SAVE_PATH   = "lunar_landing_best_weights.pth"

# ── Old Policy Network (same as training) ─────────────────────────────────────
class PolicyNetwork(nn.Module):
    def __init__(self, obs_size: int, action_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, action_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(n_episodes: int = 10, render: bool = True, temperature: float = 0.3):
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", render_mode=render_mode)

    obs_size    = env.observation_space.shape[0]
    action_size = env.action_space.n

    # ── Load checkpoint ───────────────────────────────────────────────────────
    checkpoint = torch.load(SAVE_PATH, weights_only=False)
    policy = PolicyNetwork(obs_size, action_size)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    print(f"Loaded weights  ->  trained ep={checkpoint['episode']}  "
          f"mean-100={checkpoint['best_mean_reward']:.1f}")
    print(f"Evaluating {n_episodes} episodes  (temperature={temperature})\n")

    total_rewards = []
    landed_count  = 0

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        total  = 0.0
        steps  = 0

        while True:
            obs_t = torch.tensor(obs, dtype=torch.float32)

            with torch.no_grad():
                probs = policy(obs_t)
                # Temperature-scaled selection: more decisive than pure sampling,
                # avoids the "frozen hover" issue of argmax on ambiguous states
                scaled = probs ** (1.0 / temperature)
                scaled = scaled / scaled.sum()
                action = torch.multinomial(scaled, 1).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            steps += 1

            if terminated or truncated:
                break

        total_rewards.append(total)
        landed = total > 100
        if landed:
            landed_count += 1
        status = "LANDED  ✔" if landed else "crashed ✘"
        print(f"  Ep {ep:2d}: reward = {total:8.1f}   steps = {steps:4d}   {status}")

    print(f"\n{'─'*52}")
    print(f"  Landed      : {landed_count}/{n_episodes}")
    print(f"  Mean reward : {np.mean(total_rewards):.1f}")
    print(f"  Best reward : {max(total_rewards):.1f}")
    print(f"  Worst reward: {min(total_rewards):.1f}")
    env.close()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    evaluate(
        n_episodes  = 10,
        render      = True,
        temperature = 0.3,   # lower = sharper (try 0.1), higher = softer (try 0.5)
    )