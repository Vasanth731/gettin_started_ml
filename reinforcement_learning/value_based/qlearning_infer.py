"""
infer.py — Inference script for a trained Q-Learning agent on FrozenLake-v1

Usage:
    python infer.py                    # run 5 episodes with render
    python infer.py --episodes 10      # run 10 episodes
    python infer.py --no-render        # headless evaluation over many episodes

Requires:
    q_table.npy  — produced by train.py
"""

import argparse
import numpy as np
import gymnasium as gym
import time

# ── Config ────────────────────────────────────────────────────────────────────
Q_TABLE_PATH = "frozon_lake_q_table.npy"
MAP_NAME     = "4x4"       # must match what was used during training
IS_SLIPPERY  = True
# ──────────────────────────────────────────────────────────────────────────────

ACTION_SYMBOLS = {0: "← LEFT", 1: "↓ DOWN", 2: "→ RIGHT", 3: "↑ UP"}


def run_episode(env, q_table, render=True, episode_num=1):
    """Run a single greedy episode. Returns total reward and step count."""
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    done = False

    if render:
        print(f"\n{'='*40}")
        print(f" Episode {episode_num}")
        print(f"{'='*40}")
        env.render()
        time.sleep(0.3)

    while not done:
        action = int(np.argmax(q_table[state]))   # fully greedy — no exploration

        if render:
            print(f"  State {state:>2} → Action: {ACTION_SYMBOLS[action]}")

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
        state = next_state

        if render:
            env.render()
            time.sleep(0.3)

    if render:
        outcome = "✅ GOAL REACHED" if total_reward > 0 else "❌ FELL IN HOLE"
        print(f"\n  {outcome}  (steps: {steps}, reward: {total_reward})")

    return total_reward, steps


def main():
    parser = argparse.ArgumentParser(description="FrozenLake Q-Learning Inference")
    parser.add_argument("--episodes",  type=int,  default=5,    help="Number of episodes to run")
    parser.add_argument("--no-render", action="store_true",      help="Disable visual rendering")
    args = parser.parse_args()

    # Load Q-table
    try:
        q_table = np.load(Q_TABLE_PATH)
        print(f"Loaded Q-table from '{Q_TABLE_PATH}'  shape: {q_table.shape}")
    except FileNotFoundError:
        print(f"ERROR: '{Q_TABLE_PATH}' not found. Run train.py first.")
        return

    render = not args.no_render
    render_mode = "human" if render else None

    env = gym.make(
        "FrozenLake-v1",
        map_name=MAP_NAME,
        is_slippery=IS_SLIPPERY,
        render_mode=render_mode,
    )

    rewards = []
    step_counts = []

    for ep in range(1, args.episodes + 1):
        r, s = run_episode(env, q_table, render=render, episode_num=ep)
        rewards.append(r)
        step_counts.append(s)

        if not render:
            # Simple progress for headless mode
            print(f"Episode {ep:>4}/{args.episodes} | "
                  f"{'WIN' if r > 0 else 'LOSE':4} | steps: {s}")

    env.close()

    # ── Summary statistics ─────────────────────────────────────────────────────
    wins      = sum(r > 0 for r in rewards)
    win_rate  = wins / len(rewards) * 100
    avg_steps = np.mean(step_counts)

    print(f"\n{'='*40}")
    print(f" Evaluation Summary ({args.episodes} episodes)")
    print(f"{'='*40}")
    print(f"  Wins        : {wins} / {args.episodes}")
    print(f"  Win rate    : {win_rate:.1f}%")
    print(f"  Avg steps   : {avg_steps:.1f}")
    print(f"{'='*40}")

    # ── Print best policy from Q-table ─────────────────────────────────────────
    GRID = 4 if MAP_NAME == "4x4" else 8
    frozen_map = {
        "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
        "8x8": [
            "SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF",
            "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG",
        ],
    }[MAP_NAME]
    tile_icons = {"S": "S", "F": " ", "H": "H", "G": "G"}
    arrow      = {0: "←", 1: "↓", 2: "→", 3: "↑"}

    print("\n  Greedy policy grid (H=hole, G=goal):\n")
    for row in range(GRID):
        line = "  "
        for col in range(GRID):
            state = row * GRID + col
            tile  = frozen_map[row][col]
            if tile in ("H", "G"):
                line += f"[{tile_icons[tile]}] "
            else:
                best_action = int(np.argmax(q_table[state]))
                line += f"[{arrow[best_action]}] "
        print(line)
    print()


if __name__ == "__main__":
    main()