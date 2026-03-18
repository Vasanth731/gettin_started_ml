"""
pong_ppo_infer.py  —  Inference / evaluation for the PPO-trained Pong agent
═════════════════════════════════════════════════════════════════════════════

"""
import ale_py
import gymnasium as gym
gym.register_envs(ale_py)
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
import cv2
from collections import deque

# ── Must match train constants ─────────────────────────────────────────────────
FRAME_STACK   = 4
FRAME_H       = 84
FRAME_W       = 84
VALID_ACTIONS = [0, 2, 3]
N_ACTIONS     = len(VALID_ACTIONS)
ACTION_NAMES  = {0: "NOOP   ", 2: "UP  ↑  ", 3: "DOWN ↓ "}
POLICY_PATH   = "pong_ppo_best.pth"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────


# ── Preprocessing (identical to train) ────────────────────────────────────────
def preprocess_frame(frame):
    gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cropped = gray[34:194, :]
    resized = cv2.resize(cropped, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


class FrameStack:
    def __init__(self, n=FRAME_STACK):
        self.n   = n
        self.buf = deque(maxlen=n)

    def reset(self, frame):
        proc = preprocess_frame(frame)
        for _ in range(self.n):
            self.buf.append(proc)
        return self._get()

    def step(self, frame):
        self.buf.append(preprocess_frame(frame))
        return self._get()

    def _get(self):
        return np.stack(list(self.buf), axis=0)   # (4, 84, 84)


# ── Actor-Critic Network (same as train) ──────────────────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, n_actions=N_ACTIONS):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        cnn_out = self._cnn_out_size()
        self.shared_fc = nn.Sequential(nn.Linear(cnn_out, 512), nn.ReLU())
        self.actor  = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def _cnn_out_size(self):
        dummy = torch.zeros(1, FRAME_STACK, FRAME_H, FRAME_W)
        return int(self.cnn(dummy).shape[1])

    def forward(self, x):
        feat   = self.cnn(x)
        shared = self.shared_fc(feat)
        return self.actor(shared), self.critic(shared).squeeze(-1)

    @torch.no_grad()
    def predict(self, state_np, greedy=True):
        """
        state_np : (4, 84, 84) numpy
        Returns  : (action_idx, probs, value)
        """
        t             = torch.FloatTensor(state_np).unsqueeze(0).to(DEVICE)
        logits, value = self.forward(t)
        probs         = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        if greedy:
            action_idx = int(np.argmax(probs))
        else:
            action_idx = int(np.random.choice(N_ACTIONS, p=probs))
        return action_idx, probs, float(value.cpu().item())


# ── Episode runner ─────────────────────────────────────────────────────────────
def run_episode(env, model, render, episode_num, greedy=True, delay=0.033):
    obs, _  = env.reset()
    stacker = FrameStack(FRAME_STACK)
    state   = stacker.reset(obs)

    total_reward   = 0.0
    steps          = 0
    score_agent    = 0
    score_opp      = 0
    rally          = 0
    rally_lengths  = []
    action_counts  = np.zeros(N_ACTIONS, dtype=int)
    value_trace    = []
    done           = False

    if render:
        print(f"\n{'═'*65}")
        print(f"  Episode {episode_num}  ({'Greedy' if greedy else 'Stochastic'} policy)")
        print(f"{'═'*65}")

    while not done:
        action_idx, probs, value = model.predict(state, greedy=True)
        action = VALID_ACTIONS[action_idx]
        action_counts[action_idx] += 1
        value_trace.append(value)

        obs, reward, terminated, truncated, _ = env.step(action)
        done  = terminated or truncated
        state = stacker.step(obs)

        total_reward += reward
        steps        += 1
        rally        += 1

        if reward != 0:
            if reward > 0: score_agent += 1
            else:          score_opp   += 1
            rally_lengths.append(rally)
            rally = 0

            if render:
                winner  = "Agent ✅" if reward > 0 else "CPU   ❌"
                p_str   = "  ".join(
                    f"{ACTION_NAMES[VALID_ACTIONS[i]]}:{probs[i]:.2f}"
                    for i in range(N_ACTIONS)
                )
                print(
                    f"  Step {steps:>5} | {winner} | "
                    f"Score {score_agent}-{score_opp} | "
                    f"Rally {rally_lengths[-1]:>3}s | "
                    f"V={value:+.2f} | π=[{p_str}]"
                )

        if render:
            time.sleep(delay)

    if render:
        result = "WIN 🏆" if score_agent > score_opp else "LOSS 💀" if score_agent < score_opp else "DRAW"
        print(f"\n  {result} | Score {score_agent}-{score_opp} | "
              f"Reward {total_reward:.0f} | Steps {steps} | "
              f"Avg V={np.mean(value_trace):+.2f}")

    return {
        "reward":       total_reward,
        "steps":        steps,
        "score_agent":  score_agent,
        "score_opp":    score_opp,
        "rallies":      rally_lengths,
        "action_counts": action_counts,
        "value_trace":  value_trace,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Pong PPO Inference")
    parser.add_argument("--episodes",   type=int,   default=3,           help="Number of episodes")
    parser.add_argument("--no-render",  action="store_true",             help="Headless eval")
    parser.add_argument("--stochastic", action="store_true",             help="Sample from policy")
    parser.add_argument("--delay",      type=float, default=0.033,       help="Seconds between frames (render mode)")
    parser.add_argument("--checkpoint", type=str,   default=POLICY_PATH, help="Path to .pth checkpoint")
    args = parser.parse_args()

    # ── Load model ─────────────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: '{args.checkpoint}' not found. Run pong_ppo_train.py first.")
        return

    model = ActorCritic(N_ACTIONS).to(DEVICE)

    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    # Support both raw state_dict and wrapped checkpoint dicts
    state_dict = ckpt.get("model_state", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    trained_steps = ckpt.get("global_step", "unknown")
    best_avg      = ckpt.get("best_avg",    "unknown")
    print(f"Loaded PPO checkpoint: '{args.checkpoint}'")
    print(f"  Trained steps : {trained_steps:,}" if isinstance(trained_steps, int) else f"  Trained steps : {trained_steps}")
    print(f"  Best avg-50   : {best_avg:.2f}" if isinstance(best_avg, float) else f"  Best avg-50   : {best_avg}")
    print(f"  Device        : {DEVICE}")
    print(f"  Parameters    : {sum(p.numel() for p in model.parameters()):,}\n")

    greedy      = not args.stochastic
    render      = not args.no_render
    render_mode = "human" if render else "rgb_array"

    env = gym.make("ALE/Pong-v5", render_mode=render_mode)

    all_results = []
    wins = losses = draws = 0

    for ep in range(1, args.episodes + 1):
        result = run_episode(env, model, render, ep, greedy, args.delay)
        all_results.append(result)

        sa, so = result["score_agent"], result["score_opp"]
        if sa > so:   wins   += 1
        elif sa < so: losses += 1
        else:         draws  += 1

        if not render:
            outcome = "WIN " if sa > so else "LOSS" if sa < so else "DRAW"
            print(f"  Episode {ep:>4}/{args.episodes} | {outcome} | "
                  f"Score {sa}-{so} | Reward {result['reward']:>5.0f} | "
                  f"Steps {result['steps']:>4}")

    env.close()

    # ── Summary ────────────────────────────────────────────────────────────────
    n           = len(all_results)
    rewards     = [r["reward"]  for r in all_results]
    steps_list  = [r["steps"]   for r in all_results]
    all_rallies = [l for r in all_results for l in r["rallies"]]
    agg_actions = np.sum([r["action_counts"] for r in all_results], axis=0)

    print(f"\n{'═'*60}")
    print(f"  Evaluation Summary  ({n} ep | {'greedy' if greedy else 'stochastic'})")
    print(f"{'═'*60}")
    print(f"  Wins / Losses / Draws    : {wins} / {losses} / {draws}")
    print(f"  Win rate                 : {wins/n*100:.1f}%")
    print(f"  Avg reward               : {np.mean(rewards):.2f}  ±{np.std(rewards):.2f}")
    print(f"  Avg steps / episode      : {np.mean(steps_list):.0f}")
    if all_rallies:
        print(f"  Avg rally length         : {np.mean(all_rallies):.1f} steps")
        print(f"  Longest rally            : {max(all_rallies)} steps")
    print(f"\n  Action distribution:")
    total_ac = agg_actions.sum()
    for i, a in enumerate(VALID_ACTIONS):
        pct = agg_actions[i] / total_ac * 100 if total_ac else 0
        bar = "█" * int(pct / 2)
        print(f"    {ACTION_NAMES[a]} : {agg_actions[i]:>7,}  ({pct:4.1f}%)  {bar}")
    print(f"{'═'*60}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    if n >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle(f"ALE/Pong-v5 — PPO Inference  ({n} episodes)",
                     fontsize=12, fontweight="bold")

        # Reward per episode
        colors = ["steelblue" if r > 0 else "tomato" if r < 0 else "gray" for r in rewards]
        axes[0, 0].bar(range(1, n + 1), rewards, color=colors)
        axes[0, 0].axhline(0, color="gray", linestyle="--")
        axes[0, 0].set_title("Reward per Episode")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Total Reward")

        # Win/Loss/Draw pie
        axes[0, 1].pie(
            [wins, losses, draws],
            labels=["Wins", "Losses", "Draws"],
            autopct="%1.1f%%",
            colors=["#4caf50", "#f44336", "#9e9e9e"],
            startangle=90,
        )
        axes[0, 1].set_title("Win / Loss / Draw")

        # Action distribution
        action_labels = [ACTION_NAMES[a].strip() for a in VALID_ACTIONS]
        axes[1, 0].bar(action_labels, agg_actions, color=["#95b8d1", "#809bce", "#b8e0d2"])
        axes[1, 0].set_title("Action Distribution")
        axes[1, 0].set_ylabel("Count")

        # Value trace for the last episode
        vt = all_results[-1]["value_trace"]
        if vt:
            axes[1, 1].plot(vt, color="darkorange", linewidth=1.0)
            axes[1, 1].axhline(0, color="gray", linestyle="--")
            axes[1, 1].set_title(f"Value Trace — Episode {n}")
            axes[1, 1].set_xlabel("Step")
            axes[1, 1].set_ylabel("V(s)")

        plt.tight_layout()
        plt.savefig("pong_ppo_inference_results.png", dpi=150)
        plt.show()
        print("\nInference results saved → 'pong_ppo_inference_results.png'")


if __name__ == "__main__":
    main()