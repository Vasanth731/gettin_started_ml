
# # pip install gymnasium
# # pip install "gymnasium[box2d]"

# # policy based - REINFORCE (Monte Carlo Policy Gradient)


import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ── Hyperparameters ──────────────────────────────────────────────────────────
EPISODES        = 1000          # total training episodes
GAMMA           = 0.99          # discount factor
LR              = 1e-3          # learning rate
HIDDEN_SIZE     = 128           # hidden layer size
SAVE_PATH       = "lunar_landing_best_weights_2.pth"
PRINT_EVERY     = 20            # log interval (episodes)
SOLVE_THRESHOLD = 200.0         # considered "solved" above this mean reward


# ── Policy Network ───────────────────────────────────────────────────────────
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


# ── REINFORCE update ─────────────────────────────────────────────────────────
def compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """Discounted returns, normalised for stability."""
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def update_policy(
    optimizer: optim.Optimizer,
    log_probs: list[torch.Tensor],
    returns: torch.Tensor,
) -> float:
    policy_loss = [-lp * G for lp, G in zip(log_probs, returns)]
    loss = torch.stack(policy_loss).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    env = gym.make("LunarLander-v3")
    obs_size    = env.observation_space.shape[0]   # 8
    action_size = env.action_space.n               # 4

    policy    = PolicyNetwork(obs_size, action_size)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    best_mean_reward = -float("inf")
    recent_rewards: list[float] = []

    print(f"Training on LunarLander-v3 | obs={obs_size} actions={action_size}")
    print(f"Weights will be saved to: {SAVE_PATH}\n")

    for episode in range(1, EPISODES + 1):
        obs, _ = env.reset()
        log_probs: list[torch.Tensor] = []
        rewards:   list[float]        = []

        # ── Roll out one episode ──────────────────────────────────────────────
        while True:
            obs_t  = torch.tensor(obs, dtype=torch.float32)
            probs  = policy(obs_t)
            dist   = Categorical(probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            obs, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)

            if terminated or truncated:
                break

        # ── REINFORCE update ──────────────────────────────────────────────────
        returns = compute_returns(rewards, GAMMA)
        update_policy(optimizer, log_probs, returns)

        ep_reward = sum(rewards)
        recent_rewards.append(ep_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
        mean_reward = np.mean(recent_rewards)

        # ── Save best weights ─────────────────────────────────────────────────
        if mean_reward > best_mean_reward and len(recent_rewards) == 100:
            best_mean_reward = mean_reward
            torch.save(
                {
                    "episode":          episode,
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state":  optimizer.state_dict(),
                    "best_mean_reward": best_mean_reward,
                },
                SAVE_PATH,
            )
            print(f"  ✔ New best saved  (mean-100={best_mean_reward:.1f})")

        if episode % PRINT_EVERY == 0:
            print(
                f"Ep {episode:4d}/{EPISODES}  "
                f"reward={ep_reward:8.1f}  "
                f"mean-100={mean_reward:8.1f}  "
                f"best={best_mean_reward:8.1f}"
            )

        if mean_reward >= SOLVE_THRESHOLD and len(recent_rewards) == 100:
            print(f"\n Solved at episode {episode}! Mean reward: {mean_reward:.1f}")
            break

    env.close()
    print(f"\nTraining complete. Best weights saved to: {SAVE_PATH}")


# ── Evaluation (loads saved weights) ─────────────────────────────────────────
def evaluate(n_episodes: int = 5, render: bool = True):
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    obs_size    = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy = PolicyNetwork(obs_size, action_size)
    checkpoint = torch.load(SAVE_PATH, weights_only=False)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    print(f"\nEvaluating best weights (trained to ep {checkpoint['episode']}, "
          f"mean-100={checkpoint['best_mean_reward']:.1f})")

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        total  = 0.0
        while True:
            obs_t  = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                probs  = policy(obs_t)
            action = torch.argmax(probs).item()   # greedy at eval time
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        print(f"  Eval ep {ep}: reward = {total:.1f}")

    env.close()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # train()
    evaluate()