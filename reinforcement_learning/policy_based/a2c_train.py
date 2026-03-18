"""
spaceinvaders_a2c_train.py  —  A2C (Advantage Actor-Critic) for ALE/SpaceInvaders-v5
══════════════════════════════════════════════════════════════════════════════════════

Algorithm: Synchronous Advantage Actor-Critic (A2C)
════════════════════════════════════════════════════

Requirements:
    pip install "gymnasium[atari]" ale-py torch numpy matplotlib opencv-python
"""
import ale_py
import gymnasium as gym
gym.register_envs(ale_py)
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
import cv2
from collections import deque

# ══════════════════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
NUM_ENVS          = 16          # parallel workers (A2C scales well with more)
TOTAL_TIMESTEPS   = 10_000_000  # total env steps
ROLLOUT_STEPS     = 5           # n-step TD horizon (A2C typically uses 5–20)
LEARNING_RATE     = 7e-4        # RMSProp LR (standard for A2C)
GAMMA             = 0.99        # discount factor
VALUE_COEFF       = 0.5         # critic loss weight
ENTROPY_COEFF     = 0.01        # entropy bonus weight
MAX_GRAD_NORM     = 0.5         # global gradient clip
CLIP_REWARDS      = True        # clip rewards to [−1, 1] for training stability
LR_ANNEAL         = True        # decay LR linearly to 0

FRAME_STACK       = 4
FRAME_H           = 84
FRAME_W           = 84
N_ACTIONS         = 6           # SpaceInvaders uses all 6 actions

SAVE_EVERY_STEPS  = 1_000_000
POLICY_PATH       = "si_a2c_best.pth"
CHECKPOINT_PREFIX = "si_a2c_checkpoint"
LOG_EVERY_UPDATES = 100         # print stats every N updates
# ══════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTION_NAMES = {
    0: "NOOP     ",
    1: "FIRE     ",
    2: "RIGHT    ",
    3: "LEFT     ",
    4: "RIGHTFIRE",
    5: "LEFTFIRE ",
}


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_frame(frame):
    """(210,160,3) uint8  →  (84,84) float32 [0,1]"""
    gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


class VecFrameStack:
    """
    Manages a frame buffer for N parallel environments.
    Each env maintains its own deque of FRAME_STACK recent frames.
    """
    def __init__(self, n_envs, n_stack=FRAME_STACK):
        self.n_envs  = n_envs
        self.n_stack = n_stack
        self.buf     = np.zeros((n_envs, n_stack, FRAME_H, FRAME_W), dtype=np.float32)

    def reset(self, obs_list):
        for i, obs in enumerate(obs_list):
            f = preprocess_frame(obs)
            self.buf[i] = np.stack([f] * self.n_stack)
        return self.buf.copy()

    def step(self, obs_list, dones):
        for i, (obs, done) in enumerate(zip(obs_list, dones)):
            f = preprocess_frame(obs)
            if done:
                # On episode end, fill buffer with latest frame
                self.buf[i] = np.stack([f] * self.n_stack)
            else:
                self.buf[i] = np.roll(self.buf[i], shift=-1, axis=0)
                self.buf[i, -1] = f
        return self.buf.copy()


# ── Reward normalisation via running mean/variance ─────────────────────────────
class RunningMeanStd:
    """Track running statistics for reward normalisation."""
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var  = 1.0
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var  = np.var(x)
        batch_count = x.size
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean  = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var   = (m_a + m_b + delta**2 * self.count * batch_count / total) / total
        self.count = total

    def normalise(self, x):
        return x / (np.sqrt(self.var) + 1e-8)


# ── Actor-Critic Network ───────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    """
    Nature-DQN CNN backbone shared between Actor and Critic heads.

    Input  : (B, 4, 84, 84) stacked grayscale frames
    Output : action logits (B, 6), state value (B,)
    """
    def __init__(self, n_actions=N_ACTIONS):
        super().__init__()

        # Shared feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4),   # → (32,20,20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),             # → (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),             # → (64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),
        )

        cnn_out = self._get_cnn_out()

        self.shared_fc = nn.Sequential(
            nn.Linear(cnn_out, 512),
            nn.ReLU(),
        )

        self.actor  = nn.Linear(512, n_actions)   # policy head → logits
        self.critic = nn.Linear(512, 1)            # value head  → scalar

        self._init_weights()

    def _get_cnn_out(self):
        with torch.no_grad():
            dummy = torch.zeros(1, FRAME_STACK, FRAME_H, FRAME_W)
            return self.cnn(dummy).shape[1]

    def _init_weights(self):
        """Orthogonal initialisation — standard for A2C/PPO."""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x):
        feat   = self.cnn(x)
        shared = self.shared_fc(feat)
        logits = self.actor(shared)
        value  = self.critic(shared).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs_np):
        """
        Stochastic action selection for all envs simultaneously.
        obs_np : (N, 4, 84, 84)
        Returns: actions, log_probs, values (all numpy)
        """
        obs_t          = torch.FloatTensor(obs_np).to(DEVICE)
        logits, values = self.forward(obs_t)
        dist           = torch.distributions.Categorical(logits=logits)
        actions        = dist.sample()
        log_probs      = dist.log_prob(actions)
        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
        )


# ── n-step Return Computation ──────────────────────────────────────────────────
def compute_nstep_returns(rewards, dones, last_values, gamma=GAMMA):
    """
    Bootstrap n-step returns from the end of the rollout:
      R_t = r_t + γ·r_{t+1} + … + γ^(n-t-1)·r_{n-1} + γ^(n-t)·V(s_n)·(1-done)

    rewards     : (T, N)
    dones       : (T, N)
    last_values : (N,)   — V(s_T), bootstrapped value for final state
    Returns     : (T, N) discounted returns
    """
    T, N    = rewards.shape
    returns = np.zeros_like(rewards)
    R       = last_values.copy()          # bootstrap from V(s_T)

    for t in reversed(range(T)):
        R          = rewards[t] + gamma * R * (1.0 - dones[t])
        returns[t] = R

    return returns


# ── A2C Update Step ────────────────────────────────────────────────────────────
def a2c_update(model, optimizer, obs_batch, actions_batch,
               returns_batch, global_step, total_steps):
    """
    Single gradient update on the collected rollout.

    A2C does ONE update per rollout (no epochs, no replay).
    This keeps the data strictly on-policy.
    """
    obs_t     = torch.FloatTensor(obs_batch).to(DEVICE)       # (T*N, 4, 84, 84)
    actions_t = torch.LongTensor(actions_batch).to(DEVICE)    # (T*N,)
    returns_t = torch.FloatTensor(returns_batch).to(DEVICE)   # (T*N,)

    logits, values = model.forward(obs_t)
    dist           = torch.distributions.Categorical(logits=logits)
    log_probs      = dist.log_prob(actions_t)
    entropy        = dist.entropy()

    # Advantage = Returns − V(s)  (no normalisation needed since we normalise rewards)
    advantages = returns_t - values.detach()

    # Actor (policy) loss
    actor_loss  = -(log_probs * advantages).mean()

    # Critic (value) loss — MSE between predicted value and n-step return
    critic_loss = F.mse_loss(values, returns_t)

    # Entropy bonus — encourages exploration, prevents premature convergence
    entropy_loss = -entropy.mean()

    # Combined A2C loss
    total_loss  = actor_loss + VALUE_COEFF * critic_loss + ENTROPY_COEFF * entropy_loss

    # LR annealing
    if LR_ANNEAL:
        frac = max(0.0, 1.0 - global_step / total_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = frac * LEARNING_RATE

    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    optimizer.step()

    return (
        actor_loss.item(),
        critic_loss.item(),
        entropy.mean().item(),
        advantages.abs().mean().item(),
    )


# ── Main Training Loop ─────────────────────────────────────────────────────────
def train():
    envs = [gym.make("ALE/SpaceInvaders-v5") for _ in range(NUM_ENVS)]

    model     = ActorCritic(N_ACTIONS).to(DEVICE)
    # A2C conventionally uses RMSProp (not Adam) — matches the original paper
    optimizer = optim.RMSprop(
        model.parameters(), lr=LEARNING_RATE, alpha=0.99, eps=1e-5
    )
    stacker   = VecFrameStack(NUM_ENVS, FRAME_STACK)
    rms       = RunningMeanStd()          # for reward normalisation

    param_count = sum(p.numel() for p in model.parameters())
    print("═" * 70)
    print("  A2C Agent  —  ALE/SpaceInvaders-v5")
    print("═" * 70)
    print(f"  Device          : {DEVICE}")
    print(f"  Parameters      : {param_count:,}")
    print(f"  Parallel envs   : {NUM_ENVS}")
    print(f"  Rollout steps   : {ROLLOUT_STEPS}  (n-step TD)")
    print(f"  Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"  Updates planned : {TOTAL_TIMESTEPS // (NUM_ENVS * ROLLOUT_STEPS):,}")
    print("═" * 70 + "\n")

    # Reset all environments
    obs_list = [env.reset()[0] for env in envs]
    states   = stacker.reset(obs_list)   # (N, 4, 84, 84)

    global_step     = 0
    update_count    = 0
    episode_count   = 0
    last_ckpt_step  = 0
    start_time      = time.time()
    best_avg_reward = -float("inf")

    # Per-env episode tracking
    ep_reward_buf = np.zeros(NUM_ENVS, dtype=np.float32)
    ep_len_buf    = np.zeros(NUM_ENVS, dtype=np.int32)

    # Logging buffers
    completed_rewards   = deque(maxlen=200)
    log_actor_losses    = []
    log_critic_losses   = []
    log_entropies       = []
    log_avg_adv         = []
    log_avg_rewards     = []

    # Storage for one rollout: (T, N, ...)
    mb_obs      = np.zeros((ROLLOUT_STEPS, NUM_ENVS, FRAME_STACK, FRAME_H, FRAME_W), dtype=np.float32)
    mb_actions  = np.zeros((ROLLOUT_STEPS, NUM_ENVS), dtype=np.int64)
    mb_rewards  = np.zeros((ROLLOUT_STEPS, NUM_ENVS), dtype=np.float32)
    mb_dones    = np.zeros((ROLLOUT_STEPS, NUM_ENVS), dtype=np.float32)
    mb_values   = np.zeros((ROLLOUT_STEPS, NUM_ENVS), dtype=np.float32)

    total_updates = TOTAL_TIMESTEPS // (NUM_ENVS * ROLLOUT_STEPS)

    print(f"{'Update':>8} | {'Steps':>10} | {'Episodes':>9} | "
          f"{'AvgRew':>8} | {'MaxRew':>7} | "
          f"{'ActorL':>7} | {'CritL':>7} | {'Entropy':>8} | {'SPS':>6}")
    print("─" * 86)

    for update in range(1, total_updates + 1):

        # ── Collect T-step rollout across N envs ───────────────────────────────
        for step in range(ROLLOUT_STEPS):
            actions, log_probs, values = model.act(states)

            next_obs_list = []
            raw_rewards   = []
            dones         = []

            for i, env in enumerate(envs):
                obs, reward, terminated, truncated, _ = env.step(int(actions[i]))
                done = terminated or truncated

                next_obs_list.append(obs)
                raw_rewards.append(float(reward))
                dones.append(float(done))

                ep_reward_buf[i] += reward
                ep_len_buf[i]    += 1

                if done:
                    completed_rewards.append(ep_reward_buf[i])
                    ep_reward_buf[i] = 0.0
                    ep_len_buf[i]    = 0
                    episode_count   += 1
                    reset_obs, _    = env.reset()
                    next_obs_list[i] = reset_obs

            raw_rewards_np = np.array(raw_rewards, dtype=np.float32)
            dones_np       = np.array(dones,       dtype=np.float32)

            # Reward clipping for training stability
            if CLIP_REWARDS:
                train_rewards = np.clip(raw_rewards_np, -1.0, 1.0)
            else:
                rms.update(raw_rewards_np)
                train_rewards = rms.normalise(raw_rewards_np)

            mb_obs[step]     = states
            mb_actions[step] = actions
            mb_rewards[step] = train_rewards
            mb_dones[step]   = dones_np
            mb_values[step]  = values

            states      = stacker.step(next_obs_list, dones)
            global_step += NUM_ENVS

        # Bootstrap value at end of rollout
        with torch.no_grad():
            _, last_vals = model.forward(
                torch.FloatTensor(states).to(DEVICE)
            )
            last_vals_np = last_vals.cpu().numpy()

        # Compute n-step returns
        returns = compute_nstep_returns(mb_rewards, mb_dones, last_vals_np)

        # Flatten (T, N) → (T*N,)
        flat_obs     = mb_obs.reshape(-1, FRAME_STACK, FRAME_H, FRAME_W)
        flat_actions = mb_actions.reshape(-1)
        flat_returns = returns.reshape(-1)

        # ── Single A2C gradient update ─────────────────────────────────────────
        al, cl, ent, avg_adv = a2c_update(
            model, optimizer,
            flat_obs, flat_actions, flat_returns,
            global_step, TOTAL_TIMESTEPS,
        )
        update_count += 1

        log_actor_losses.append(al)
        log_critic_losses.append(cl)
        log_entropies.append(ent)
        log_avg_adv.append(avg_adv)

        avg_rew = np.mean(completed_rewards) if completed_rewards else float("nan")
        max_rew = np.max(completed_rewards)  if completed_rewards else float("nan")
        log_avg_rewards.append(avg_rew)

        # ── Logging ────────────────────────────────────────────────────────────
        if update % LOG_EVERY_UPDATES == 0:
            sps = int(global_step / (time.time() - start_time))
            print(
                f"{update:>8,} | {global_step:>10,} | {episode_count:>9,} | "
                f"{avg_rew:>8.1f} | {max_rew:>7.1f} | "
                f"{al:>7.4f} | {cl:>7.4f} | {ent:>8.4f} | {sps:>6}"
            )

        # ── Save best model ────────────────────────────────────────────────────
        if len(completed_rewards) >= 50 and avg_rew > best_avg_reward:
            best_avg_reward = avg_rew
            torch.save({
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "global_step":     global_step,
                "best_avg_reward": best_avg_reward,
                "n_actions":       N_ACTIONS,
            }, POLICY_PATH)
            print(f"  ★ New best avg-200: {best_avg_reward:.1f}  →  '{POLICY_PATH}'")

        # ── Periodic checkpoint ────────────────────────────────────────────────
        if global_step - last_ckpt_step >= SAVE_EVERY_STEPS:
            ckpt = f"{CHECKPOINT_PREFIX}_{global_step:08d}.pth"
            torch.save({
                "model_state": model.state_dict(),
                "global_step": global_step,
            }, ckpt)
            last_ckpt_step = global_step
            print(f"  ✔ Checkpoint  →  '{ckpt}'")

    for env in envs:
        env.close()

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/3600:.2f} h  |  Best avg-200: {best_avg_reward:.1f}")

    # ── Training diagnostics plot ──────────────────────────────────────────────
    _plot_training(log_avg_rewards, log_actor_losses, log_critic_losses,
                   log_entropies,   log_avg_adv,       LOG_EVERY_UPDATES)


def _plot_training(rewards, actor_l, critic_l, entropies, avg_adv, stride):
    x = np.arange(len(rewards)) * stride   # x-axis in updates

    def smooth(arr, w=20):
        if len(arr) < w:
            return arr
        return np.convolve(arr, np.ones(w) / w, mode="valid")

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("ALE/SpaceInvaders-v5 — A2C Training Diagnostics",
                 fontsize=13, fontweight="bold")

    # Avg reward
    axes[0, 0].plot(smooth(rewards), color="steelblue", linewidth=1.5)
    axes[0, 0].set_title("Avg Reward (last 200 eps)")
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].axhline(0, color="gray", linestyle="--", linewidth=0.8)

    # Actor loss
    axes[0, 1].plot(smooth(actor_l), color="tomato", linewidth=1.2)
    axes[0, 1].set_title("Actor (Policy) Loss")
    axes[0, 1].set_ylabel("Loss")

    # Critic loss
    axes[1, 0].plot(smooth(critic_l), color="seagreen", linewidth=1.2)
    axes[1, 0].set_title("Critic (Value) Loss")
    axes[1, 0].set_ylabel("MSE Loss")

    # Entropy
    axes[1, 1].plot(smooth(entropies), color="darkorange", linewidth=1.2)
    axes[1, 1].set_title("Policy Entropy")
    axes[1, 1].set_ylabel("Entropy (nats)")

    # Average absolute advantage
    axes[2, 0].plot(smooth(avg_adv), color="mediumpurple", linewidth=1.2)
    axes[2, 0].set_title("Mean |Advantage|")
    axes[2, 0].set_ylabel("|A_t|")
    axes[2, 0].set_xlabel("Update")

    # Reward histogram
    axes[2, 1].hist(rewards, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    axes[2, 1].set_title("Reward Distribution (all logged points)")
    axes[2, 1].set_xlabel("Avg Reward")
    axes[2, 1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig("si_a2c_training_curve.png", dpi=150)
    plt.show()
    print("Training curve saved → 'si_a2c_training_curve.png'")


if __name__ == "__main__":
    train()