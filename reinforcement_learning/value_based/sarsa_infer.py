"""
taxi_infer.py — Inference script for a SARSA-trained agent on Taxi-v3

Usage:
    python taxi_infer.py                        # 5 rendered episodes
    python taxi_infer.py --episodes 10          # 10 rendered episodes
    python taxi_infer.py --episodes 500 --no-render   # fast batch eval
    python taxi_infer.py --delay 0.5            # slow down rendering (seconds)

Requires:
    taxi_q_table.npy  — produced by taxi_train.py
"""

import argparse
import time
import numpy as np
import gymnasium as gym
from collections import defaultdict

# ── Config ─────────────────────────────────────────────────────────────────────
Q_TABLE_PATH = "taxi_q_table.npy"
# ──────────────────────────────────────────────────────────────────────────────

ACTION_NAMES = {
    0: "↓  Move South",
    1: "↑  Move North",
    2: "→  Move East",
    3: "←  Move West",
    4: "⬆  PICKUP",
    5: "⬇  DROPOFF",
}

PASSENGER_LOCS = {0: "Red", 1: "Green", 2: "Yellow", 3: "Blue", 4: "In Taxi"}
DESTINATIONS   = {0: "Red", 1: "Green", 2: "Yellow", 3: "Blue"}


def decode_state(state_int):
    """
    Decode Taxi-v3 integer state into human-readable components.
    Encoding: ((taxi_row * 5 + taxi_col) * 5 + passenger_loc) * 4 + destination
    """
    dest        = state_int % 4;         state_int //= 4
    pass_loc    = state_int % 5;         state_int //= 5
    taxi_col    = state_int % 5;         state_int //= 5
    taxi_row    = state_int
    return taxi_row, taxi_col, pass_loc, dest


def greedy_action(q_table, state, action_mask=None):
    """Pure greedy policy — no exploration."""
    if action_mask is not None:
        valid = np.where(action_mask == 1)[0]
        return int(valid[np.argmax(q_table[state][valid])])
    return int(np.argmax(q_table[state]))


def run_episode(env, q_table, render=True, episode_num=1, delay=0.4):
    """
    Run one greedy episode.
    Returns: (total_reward, steps, penalties, success)
    """
    state, info = env.reset()
    action_mask = info.get("action_mask", None)

    total_reward = 0
    penalties    = 0
    steps        = 0
    done         = False

    if render:
        print(f"\n{'═'*52}")
        print(f"  Episode {episode_num}")
        print(f"{'═'*52}")
        _print_state_info(state, step=0)
        env.render()
        time.sleep(delay)

    while not done:
        action = greedy_action(q_table, state, action_mask)

        next_state, reward, terminated, truncated, next_info = env.step(action)
        done        = terminated or truncated
        action_mask = next_info.get("action_mask", None)

        if reward == -10:
            penalties += 1

        total_reward += reward
        steps        += 1
        state         = next_state

        if render:
            outcome_tag = ""
            if reward == 20:
                outcome_tag = "  ✅ +20 PASSENGER DELIVERED!"
            elif reward == -10:
                outcome_tag = "  ❌ -10 ILLEGAL ACTION"

            print(f"\n  Step {steps:>3} | {ACTION_NAMES[action]}{outcome_tag}")
            _print_state_info(state, step=steps)
            env.render()
            time.sleep(delay)

    success = total_reward > 0  # delivered passenger = positive net reward

    if render:
        result = "✅ SUCCESS" if success else "❌ FAILED (timeout)"
        print(f"\n  {result}  |  Total reward: {total_reward}  |  "
              f"Steps: {steps}  |  Illegal moves: {penalties}")

    return total_reward, steps, penalties, success


def _print_state_info(state, step):
    """Print decoded state info in a human-friendly way."""
    taxi_row, taxi_col, pass_loc, dest = decode_state(state)
    pass_str = PASSENGER_LOCS.get(pass_loc, "?")
    dest_str = DESTINATIONS.get(dest, "?")
    print(f"  State {state:>3} | Taxi: ({taxi_row},{taxi_col}) | "
          f"Passenger: {pass_str:8} | Destination: {dest_str}")


def main():
    parser = argparse.ArgumentParser(description="Taxi-v3 SARSA Inference")
    parser.add_argument("--episodes",  type=int,   default=5,   help="Number of episodes")
    parser.add_argument("--no-render", action="store_true",      help="Disable rendering (batch eval)")
    parser.add_argument("--delay",     type=float, default=0.4,  help="Seconds between steps when rendering")
    args = parser.parse_args()

    # ── Load Q-table ───────────────────────────────────────────────────────────
    try:
        q_table = np.load(Q_TABLE_PATH)
        print(f"Loaded Q-table: '{Q_TABLE_PATH}'  shape: {q_table.shape}")
    except FileNotFoundError:
        print(f"ERROR: '{Q_TABLE_PATH}' not found. Run taxi_train.py first.")
        return

    render      = not args.no_render
    render_mode = "human" if render else None

    env = gym.make("Taxi-v3", render_mode=render_mode)

    all_rewards   = []
    all_steps     = []
    all_penalties = []
    all_successes = []

    for ep in range(1, args.episodes + 1):
        r, s, p, ok = run_episode(
            env, q_table,
            render=render,
            episode_num=ep,
            delay=args.delay,
        )
        all_rewards.append(r)
        all_steps.append(s)
        all_penalties.append(p)
        all_successes.append(ok)

        if not render:
            status = "SUCCESS" if ok else "FAILED "
            print(f"  Episode {ep:>4}/{args.episodes} | {status} | "
                  f"Reward: {r:>5} | Steps: {s:>3} | Penalties: {p}")

    env.close()

    # ── Summary ────────────────────────────────────────────────────────────────
    n          = len(all_rewards)
    wins       = sum(all_successes)
    win_rate   = wins / n * 100
    avg_reward = np.mean(all_rewards)
    avg_steps  = np.mean(all_steps)
    avg_pen    = np.mean(all_penalties)

    print(f"\n{'═'*52}")
    print(f"  Evaluation Summary  ({n} episodes)")
    print(f"{'═'*52}")
    print(f"  Success rate      : {wins}/{n}  ({win_rate:.1f}%)")
    print(f"  Avg total reward  : {avg_reward:.2f}")
    print(f"  Avg steps/episode : {avg_steps:.1f}")
    print(f"  Avg illegal moves : {avg_pen:.2f}")
    print(f"{'═'*52}")

    if wins < n:
        print(f"\n  ℹ  {n - wins} episode(s) hit the 200-step truncation limit.")
        print("     Try more training episodes if success rate is low.")

    # ── Per-action usage breakdown ─────────────────────────────────────────────
    print("\n  Greedy policy — most preferred action per state (sample of 20):")
    print(f"  {'State':>6} | {'Taxi':>8} | {'Passenger':>10} | {'Dest':>6} | Best Action")
    print("  " + "-" * 58)
    sampled_states = np.random.choice(500, size=20, replace=False)
    for s in sorted(sampled_states):
        best_a = int(np.argmax(q_table[s]))
        r, c, pl, d = decode_state(s)
        print(f"  {s:>6} | ({r},{c})     | {PASSENGER_LOCS[pl]:>10} | "
              f"{DESTINATIONS[d]:>6} | {ACTION_NAMES[best_a]}")


if __name__ == "__main__":
    main()