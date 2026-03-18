"""
train.py — Q-Learning training script for FrozenLake-v1 (Gymnasium)

Usage:
    python train.py

Outputs:
    q_table.npy  — saved Q-table for use in infer.py
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPISODES        = 20_000   # total training episodes
LEARNING_RATE   = 0.8      # alpha: how fast we update Q-values
DISCOUNT        = 0.95     # gamma: importance of future rewards
EPSILON_START   = 1.0      # start fully random
EPSILON_END     = 0.01     # minimum exploration rate
EPSILON_DECAY   = 0.0005   # decay per episode
MAP_NAME        = "4x4"    # "4x4" or "8x8"
IS_SLIPPERY     = True     # True = stochastic env (harder); False = deterministic
Q_TABLE_PATH    = "frozon_lake_q_table.npy"
# ──────────────────────────────────────────────────────────────────────────────


def epsilon_greedy(q_table, state, epsilon):
    """Choose action via ε-greedy policy."""
    if np.random.random() < epsilon:
        return env.action_space.sample()          # explore
    return int(np.argmax(q_table[state]))         # exploit


def decay_epsilon(epsilon):
    return max(EPSILON_END, epsilon - EPSILON_DECAY)


if __name__ == "__main__":
    env = gym.make(
        "FrozenLake-v1",
        map_name=MAP_NAME,
        is_slippery=IS_SLIPPERY,
    )

    n_states  = env.observation_space.n   # 16 for 4x4
    n_actions = env.action_space.n        # 4

    # Initialise Q-table to zeros
    q_table = np.zeros((n_states, n_actions))

    epsilon = EPSILON_START
    rewards_per_episode = []

    print(f"Training on FrozenLake-v1 ({MAP_NAME}, slippery={IS_SLIPPERY})")
    print(f"States: {n_states}  |  Actions: {n_actions}\n")

    for episode in range(1, EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = epsilon_greedy(q_table, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-learning update rule:
            # Q(s,a) ← Q(s,a) + α [ r + γ·max_a' Q(s',a') − Q(s,a) ]
            best_next = np.max(q_table[next_state])
            td_target = reward + DISCOUNT * best_next * (not terminated)
            td_error  = td_target - q_table[state, action]
            q_table[state, action] += LEARNING_RATE * td_error

            state = next_state
            total_reward += reward

        epsilon = decay_epsilon(epsilon)
        rewards_per_episode.append(total_reward)

        # Progress logging every 1000 episodes
        if episode % 1000 == 0:
            recent_win_rate = np.mean(rewards_per_episode[-1000:]) * 100
            print(
                f"Episode {episode:>6} / {EPISODES} | "
                f"Win rate (last 1k): {recent_win_rate:5.1f}% | "
                f"ε = {epsilon:.4f}"
            )

    env.close()

    # Save Q-table
    np.save(Q_TABLE_PATH, q_table)
    print(f"\nQ-table saved to '{Q_TABLE_PATH}'")

    # ── Plot training curve ────────────────────────────────────────────────────
    window = 500
    smoothed = np.convolve(rewards_per_episode, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(10, 4))
    plt.plot(smoothed, color="steelblue", linewidth=1.5)
    plt.xlabel("Episode")
    plt.ylabel(f"Win rate (rolling {window})")
    plt.title(f"FrozenLake Q-Learning — {MAP_NAME}, slippery={IS_SLIPPERY}")
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    plt.show()
    print("Training curve saved to 'training_curve.png'")

    # ── Print final Q-table ────────────────────────────────────────────────────
    action_labels = ["←", "↓", "→", "↑"]
    print("\nFinal Q-table (rows=states, cols=actions):")
    header = "State | " + "  ".join(f"{a:>6}" for a in action_labels)
    print(header)
    print("-" * len(header))
    for s in range(n_states):
        best = np.argmax(q_table[s])
        row = f"  {s:>3} | " + "  ".join(
            f"{'▶' if i == best else ' '}{q_table[s, i]:5.3f}"
            for i in range(n_actions)
        )
        print(row)