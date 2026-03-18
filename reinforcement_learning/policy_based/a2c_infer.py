"""
spaceinvaders_a2c_infer.py  —  Inference / evaluation for A2C-trained SpaceInvaders agent
══════════════════════════════════════════════════════════════════════════════════════════

Requirements:
    pip install "gymnasium[atari]" ale-py torch numpy matplotlib opencv-python
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
FRAME_STACK  = 4
FRAME_H      = 84
FRAME_W      = 84
N_ACTIONS    = 6
POLICY_PATH  = "si_a2c_best.pth"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTION_NAMES = {
    0: "NOOP     ",
    1: "FIRE     ",
    2: "RIGHT    ",
    3: "LEFT     ",
    4: "RIGHTFIRE",
    5: "LEFTFIRE ",
}

# Point values per enemy row (Space Invaders scoring)
# Top rows are worth more; used for display only
ENEMY_POINTS = [30, 20, 20, 10, 10]
# ──────────────────────────────────────────────────────────────────────────────


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_frame(frame):
    gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


class FrameStack:
    def __init__(self, n=FRAME_STACK):
        self.n   = n
        self.buf = deque(maxlen=n)

    def reset(self, frame):
        f = preprocess_frame(frame)
        for _ in range(self.n):
            self.buf.append(f)
        return self._get()

    def step(self, frame):
        self.buf.append(preprocess_frame(frame))
        return self._get()

    def _get(self):
        return np.stack(list(self.buf), axis=0)   # (4, 84, 84)


# ── Actor-Critic Network (identical to train) ──────────────────────────────────
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
        cnn_out = self._get_cnn_out()
        self.shared_fc = nn.Sequential(nn.Linear(cnn_out, 512), nn.ReLU())
        self.actor  = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def _get_cnn_out(self):
        with torch.no_grad():
            dummy = torch.zeros(1, FRAME_STACK, FRAME_H, FRAME_W)
            return self.cnn(dummy).shape[1]

    def forward(self, x):
        feat   = self.cnn(x)
        shared = self.shared_fc(feat)
        return self.actor(shared), self.critic(shared).squeeze(-1)

    @torch.no_grad()
    def predict(self, state_np, greedy=True):
        """
        state_np : (4, 84, 84) numpy
        Returns  : action (int), probs (6,), value (float)
        """
        t              = torch.FloatTensor(state_np).unsqueeze(0).to(DEVICE)
        logits, value  = self.forward(t)
        probs          = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        if greedy:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(N_ACTIONS, p=probs))
        return action, probs, float(value.item())


# ── Episode Statistics Tracker ─────────────────────────────────────────────────
class EpisodeStats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_reward  = 0.0
        self.steps         = 0
        self.kills         = 0         # reward events with r > 0
        self.action_counts = np.zeros(N_ACTIONS, dtype=int)
        self.value_trace   = []
        self.reward_trace  = []
        self.score_events  = []        # (step, reward) when a kill happens

    def update(self, action, reward, value):
        self.total_reward += reward
        self.steps        += 1
        self.action_counts[action] += 1
        self.value_trace.append(value)
        self.reward_trace.append(reward)
        if reward > 0:
            self.kills    += 1
            self.score_events.append((self.steps, reward))


# ── Episode Runner ─────────────────────────────────────────────────────────────
def run_episode(env, model, render, episode_num, greedy=True, delay=0.033):
    obs, _  = env.reset()
    stacker = FrameStack(FRAME_STACK)
    state   = stacker.reset(obs)
    stats   = EpisodeStats()
    done    = False

    if render:
        print(f"\n{'═'*68}")
        print(f"  Episode {episode_num}  |  {'Greedy' if greedy else 'Stochastic'} policy")
        print(f"{'═'*68}")

    while not done:
        action, probs, value = model.predict(state, greedy=greedy)

        obs, reward, terminated, truncated, _ = env.step(action)
        done  = terminated or truncated
        state = stacker.step(obs)

        stats.update(action, reward, value)

        if render and reward > 0:
            # Only print on scoring events to keep output readable
            top_actions = np.argsort(probs)[::-1][:3]
            p_str = "  ".join(
                f"{ACTION_NAMES[a].strip()}:{probs[a]:.2f}" for a in top_actions
            )
            print(
                f"  Step {stats.steps:>5} | +{reward:>2.0f} pts  "
                f"[Total: {stats.total_reward:>5.0f}]  "
                f"V={value:+6.1f}  |  top-π: [{p_str}]"
            )

        if render:
            time.sleep(delay)

    if render:
        print(f"\n  Episode done | Score: {stats.total_reward:.0f} | "
              f"Kills: {stats.kills} | Steps: {stats.steps} | "
              f"Avg V: {np.mean(stats.value_trace):+.2f}")

    return stats


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SpaceInvaders A2C Inference")
    parser.add_argument("--episodes",   type=int,   default=3,           help="Episodes to run")
    parser.add_argument("--no-render",  action="store_true",             help="Headless batch eval")
    parser.add_argument("--stochastic", action="store_true",             help="Sample from policy")
    parser.add_argument("--delay",      type=float, default=0.033,       help="Delay between frames (s)")
    parser.add_argument("--checkpoint", type=str,   default=POLICY_PATH, help="Path to .pth file")
    args = parser.parse_args()

    # ── Load model ─────────────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: '{args.checkpoint}' not found. Run spaceinvaders_a2c_train.py first.")
        return

    ckpt      = torch.load(args.checkpoint, map_location=DEVICE)
    n_actions = ckpt.get("n_actions", N_ACTIONS)
    model     = ActorCritic(n_actions).to(DEVICE)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()

    trained_steps = ckpt.get("global_step",     "N/A")
    best_avg      = ckpt.get("best_avg_reward",  "N/A")
    print("═" * 60)
    print("  A2C Inference  —  ALE/SpaceInvaders-v5")
    print("═" * 60)
    print(f"  Checkpoint    : {args.checkpoint}")
    if isinstance(trained_steps, int):
        print(f"  Trained steps : {trained_steps:,}")
    if isinstance(best_avg, float):
        print(f"  Best avg-200  : {best_avg:.1f}")
    print(f"  Device        : {DEVICE}")
    print(f"  Policy mode   : {'Stochastic' if args.stochastic else 'Greedy'}")
    print(f"  Parameters    : {sum(p.numel() for p in model.parameters()):,}")
    print("═" * 60 + "\n")

    greedy      = not args.stochastic
    render      = not args.no_render
    render_mode = "human" if render else "rgb_array"

    env          = gym.make("ALE/SpaceInvaders-v5", render_mode=render_mode)
    all_stats    = []

    for ep in range(1, args.episodes + 1):
        s = run_episode(env, model, render, ep, greedy, args.delay)
        all_stats.append(s)

        if not render:
            print(f"  Episode {ep:>4}/{args.episodes} | "
                  f"Score: {s.total_reward:>6.0f} | "
                  f"Kills: {s.kills:>3} | "
                  f"Steps: {s.steps:>4} | "
                  f"Avg V: {np.mean(s.value_trace):+.1f}")

    env.close()

    # ── Aggregate stats ────────────────────────────────────────────────────────
    n            = len(all_stats)
    rewards      = [s.total_reward for s in all_stats]
    kills        = [s.kills        for s in all_stats]
    steps_list   = [s.steps        for s in all_stats]
    agg_actions  = np.sum([s.action_counts for s in all_stats], axis=0)
    total_acts   = agg_actions.sum()

    print(f"\n{'═'*60}")
    print(f"  Evaluation Summary  ({n} episode{'s' if n > 1 else ''})")
    print(f"{'═'*60}")
    print(f"  Avg score         : {np.mean(rewards):>8.1f}  ±{np.std(rewards):.1f}")
    print(f"  Max score         : {np.max(rewards):>8.1f}")
    print(f"  Min score         : {np.min(rewards):>8.1f}")
    print(f"  Avg kills/episode : {np.mean(kills):>8.1f}")
    print(f"  Avg steps/episode : {np.mean(steps_list):>8.0f}")
    print(f"\n  Action distribution:")
    for i in range(N_ACTIONS):
        pct = agg_actions[i] / total_acts * 100 if total_acts else 0
        bar = "█" * int(pct / 2)
        print(f"    [{i}] {ACTION_NAMES[i]} : {agg_actions[i]:>8,}  ({pct:5.1f}%)  {bar}")
    print(f"{'═'*60}")

    # ── Score milestones breakdown ─────────────────────────────────────────────
    thresholds = [100, 250, 500, 750, 1000, 1500]
    print(f"\n  Score milestone distribution:")
    for lo, hi in zip([0] + thresholds, thresholds + [float("inf")]):
        count = sum(lo <= r < hi for r in rewards)
        label = f"[{lo:.0f} – {hi:.0f})" if hi != float("inf") else f"[{lo:.0f}+)"
        bar   = "█" * count
        print(f"    {label:>15} : {count:>3} ep  {bar}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    if n >= 2:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle(
            f"ALE/SpaceInvaders-v5 — A2C Inference  ({n} episodes, "
            f"{'greedy' if greedy else 'stochastic'})",
            fontsize=12, fontweight="bold",
        )

        # 1. Score per episode
        colors = plt.cm.RdYlGn(
            (np.array(rewards) - min(rewards)) /
            (max(rewards) - min(rewards) + 1e-8)
        )
        axes[0, 0].bar(range(1, n + 1), rewards, color=colors)
        axes[0, 0].axhline(np.mean(rewards), color="blue",
                           linestyle="--", linewidth=1.2, label=f"Mean={np.mean(rewards):.0f}")
        axes[0, 0].set_title("Score per Episode")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].legend(fontsize=8)

        # 2. Score distribution (histogram)
        axes[0, 1].hist(rewards, bins=min(20, n), color="steelblue",
                        edgecolor="white", alpha=0.85)
        axes[0, 1].axvline(np.mean(rewards), color="red",
                           linestyle="--", linewidth=1.2, label=f"Mean={np.mean(rewards):.0f}")
        axes[0, 1].set_title("Score Distribution")
        axes[0, 1].set_xlabel("Score")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].legend(fontsize=8)

        # 3. Kills per episode
        axes[0, 2].bar(range(1, n + 1), kills, color="tomato", alpha=0.85)
        axes[0, 2].axhline(np.mean(kills), color="darkred",
                           linestyle="--", linewidth=1.2, label=f"Mean={np.mean(kills):.1f}")
        axes[0, 2].set_title("Enemy Kills per Episode")
        axes[0, 2].set_xlabel("Episode")
        axes[0, 2].set_ylabel("Kills")
        axes[0, 2].legend(fontsize=8)

        # 4. Action distribution (bar)
        act_labels = [ACTION_NAMES[i].strip() for i in range(N_ACTIONS)]
        act_pcts   = agg_actions / total_acts * 100 if total_acts else agg_actions
        bar_colors = ["#95b8d1", "#f4a261", "#2a9d8f", "#e9c46a", "#264653", "#e76f51"]
        axes[1, 0].bar(act_labels, act_pcts, color=bar_colors)
        axes[1, 0].set_title("Action Distribution (%)")
        axes[1, 0].set_ylabel("Usage %")
        axes[1, 0].tick_params(axis="x", rotation=20)

        # 5. Value trace of the last episode
        vt = all_stats[-1].value_trace
        axes[1, 1].plot(vt, color="darkorange", linewidth=0.8, alpha=0.9)
        axes[1, 1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
        # Overlay kill events as vertical markers
        for step, r in all_stats[-1].score_events:
            axes[1, 1].axvline(step, color="green", alpha=0.4, linewidth=0.8)
        axes[1, 1].set_title(f"Value Trace — Episode {n}  (green=kill)")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("V(s)")

        # 6. Cumulative reward trace of the last episode
        cum_rew = np.cumsum(all_stats[-1].reward_trace)
        axes[1, 2].plot(cum_rew, color="seagreen", linewidth=1.2)
        axes[1, 2].set_title(f"Cumulative Score — Episode {n}")
        axes[1, 2].set_xlabel("Step")
        axes[1, 2].set_ylabel("Cumulative Score")

        plt.tight_layout()
        plt.savefig("si_a2c_inference_results.png", dpi=150)
        plt.show()
        print("\nInference results saved → 'si_a2c_inference_results.png'")


if __name__ == "__main__":
    main()