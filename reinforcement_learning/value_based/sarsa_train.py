"""
taxi_train.py — SARSA training script for Taxi-v3 (Gymnasium)

SARSA is an ON-POLICY TD method. Unlike Q-Learning (which bootstraps from the
greedy next action), SARSA updates using the ACTUAL next action chosen by the
current ε-greedy policy:

    Q(s,a) ← Q(s,a) + α [ r + γ·Q(s',a') − Q(s,a) ]
                                    ↑
                        a' is sampled from π (not max)

This makes SARSA more conservative — it accounts for exploration noise during
learning, which is especially useful in environments with heavy step penalties
like Taxi (−1 per step, −10 for illegal pickup/dropoff).

Usage:
    python taxi_train.py

Outputs:
    taxi_q_table.npy       — saved Q-table for use in taxi_infer.py
    taxi_training_curve.png — reward & penalty curves over training
"""

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPISODES        = 50_000   # total training episodes
LEARNING_RATE   = 0.7      # alpha
DISCOUNT        = 0.95     # gamma
EPSILON_START   = 1.0      # start fully exploratory
EPSILON_END     = 0.01     # floor
EPSILON_DECAY   = 0.00005  # linear decay per step (not per episode)
Q_TABLE_PATH    = "taxi_q_table.npy"
LOG_INTERVAL    = 5_000    # print stats every N episodes
# ──────────────────────────────────────────────────────────────────────────────

# Taxi-v3 specifics
N_STATES  = 500   # 25 taxi pos × 5 passenger loc × 4 destinations
N_ACTIONS = 6     # south, north, east, west, pickup, dropoff

ACTION_NAMES = ["↓ South", "↑ North", "→ East", "← West", "⬆ Pickup", "⬇ Dropoff"]


def epsilon_greedy(q_table, state, epsilon, action_mask=None):
    """
    ε-greedy policy with optional action masking.
    action_mask (from env info) avoids no-op moves into walls, etc.
    """
    if np.random.random() < epsilon:
        if action_mask is not None:
            # Sample only from actions that actually change state
            valid = np.where(action_mask == 1)[0]
            return int(np.random.choice(valid))
        return np.random.randint(N_ACTIONS)

    if action_mask is not None:
        valid = np.where(action_mask == 1)[0]
        return int(valid[np.argmax(q_table[state][valid])])

    return int(np.argmax(q_table[state]))


if __name__ == "__main__":
    env = gym.make("Taxi-v3")

    # Q-table: all zeros — shape (500, 6)
    q_table = np.zeros((N_STATES, N_ACTIONS))

    epsilon = EPSILON_START
    total_steps = 0

    episode_rewards   = []
    episode_penalties = []
    episode_steps     = []

    print("Training Taxi-v3 with SARSA")
    print(f"States: {N_STATES}  |  Actions: {N_ACTIONS}\n")

    for episode in range(1, EPISODES + 1):

        state, info = env.reset()
        action_mask = info.get("action_mask", None)

        # SARSA: choose first action BEFORE the loop (we need (s,a) pair)
        action = epsilon_greedy(q_table, state, epsilon, action_mask)

        total_reward = 0
        penalties    = 0
        steps        = 0
        done         = False

        while not done:
            next_state, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            # Count illegal move penalties
            if reward == -10:
                penalties += 1

            # ── SARSA Core ─────────────────────────────────────────────────────
            # Choose NEXT action from the CURRENT policy (on-policy)
            next_action_mask = next_info.get("action_mask", None)
            next_action = epsilon_greedy(q_table, next_state, epsilon, next_action_mask)

            # SARSA update:  Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') − Q(s,a)]
            td_target = reward + DISCOUNT * q_table[next_state, next_action] * (not terminated)
            td_error  = td_target - q_table[state, action]
            q_table[state, action] += LEARNING_RATE * td_error
            # ──────────────────────────────────────────────────────────────────

            state       = next_state
            action      = next_action   # ← key difference vs Q-learning
            action_mask = next_action_mask

            total_reward += reward
            steps        += 1
            total_steps  += 1

            # Decay epsilon at every step for smooth annealing
            epsilon = max(EPSILON_END, epsilon - EPSILON_DECAY)

        episode_rewards.append(total_reward)
        episode_penalties.append(penalties)
        episode_steps.append(steps)

        if episode % LOG_INTERVAL == 0:
            recent = slice(-LOG_INTERVAL, None)
            avg_r  = np.mean(episode_rewards[recent])
            avg_p  = np.mean(episode_penalties[recent])
            avg_s  = np.mean(episode_steps[recent])
            print(
                f"Episode {episode:>6}/{EPISODES} | "
                f"Avg reward: {avg_r:7.2f} | "
                f"Avg penalties: {avg_p:.2f} | "
                f"Avg steps: {avg_s:.1f} | "
                f"ε = {epsilon:.5f}"
            )

    env.close()

    # Save Q-table
    np.save(Q_TABLE_PATH, q_table)
    print(f"\nQ-table saved → '{Q_TABLE_PATH}'")

    # ── Training curves ────────────────────────────────────────────────────────
    window = 1000
    def smooth(x):
        return np.convolve(x, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig.suptitle("Taxi-v3 — SARSA Training", fontsize=14, fontweight="bold")

    axes[0].plot(smooth(episode_rewards),   color="steelblue",  linewidth=1.2)
    axes[0].set_ylabel("Avg Total Reward")
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)

    axes[1].plot(smooth(episode_penalties), color="tomato",     linewidth=1.2)
    axes[1].set_ylabel("Avg Penalties")

    axes[2].plot(smooth(episode_steps),     color="seagreen",   linewidth=1.2)
    axes[2].set_ylabel("Avg Steps")
    axes[2].set_xlabel(f"Episode (smoothed over {window})")

    plt.tight_layout()
    plt.savefig("taxi_training_curve.png", dpi=150)
    plt.show()
    print("Training curve saved → 'taxi_training_curve.png'")

    # ── Q-table stats ──────────────────────────────────────────────────────────
    print(f"\nQ-table stats:")
    print(f"  Non-zero entries : {np.count_nonzero(q_table)} / {q_table.size}")
    print(f"  Q-value range    : [{q_table.min():.3f}, {q_table.max():.3f}]")
    print(f"\nTop 5 highest Q-values learned:")
    flat_idx = np.argsort(q_table.ravel())[::-1][:5]
    for idx in flat_idx:
        s, a = divmod(idx, N_ACTIONS)
        print(f"  State {s:>3}, Action '{ACTION_NAMES[a]}' → Q = {q_table[s, a]:.4f}")