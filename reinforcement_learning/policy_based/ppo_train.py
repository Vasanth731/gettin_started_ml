"""
pong_ppo_train.py  —  PPO (Proximal Policy Optimization) for ALE/Pong-v5
═══════════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════
#  HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════
NUM_ENVS          = 8          # parallel environments for data collection
TOTAL_TIMESTEPS   = 5_000_000  # total env steps across all envs
ROLLOUT_STEPS     = 128        # steps collected per env before each update
MINI_BATCH_SIZE   = 256        # minibatch size for gradient updates
PPO_EPOCHS        = 4          # number of passes over each rollout
LEARNING_RATE     = 2.5e-4
GAMMA             = 0.99       # discount factor
GAE_LAMBDA        = 0.95       # GAE smoothing (0=TD, 1=MC)
CLIP_EPS          = 0.1        # PPO clip range ε
VALUE_COEFF       = 0.5        # c1: value loss weight
ENTROPY_COEFF     = 0.01       # c2: entropy bonus weight
MAX_GRAD_NORM     = 0.5        # gradient clipping
LR_ANNEAL         = True       # linearly anneal LR to 0

FRAME_STACK       = 4
FRAME_H           = 84
FRAME_W           = 84

VALID_ACTIONS     = [0, 2, 3]  # NOOP, UP, DOWN
N_ACTIONS         = len(VALID_ACTIONS)

SAVE_EVERY_STEPS  = 500_000    # checkpoint every N timesteps
POLICY_PATH       = "pong_ppo_best.pth"
CHECKPOINT_PREFIX = "pong_ppo_checkpoint"
# ══════════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_frame(frame):
    """(210,160,3) uint8  →  (84,84) float32 in [0,1]"""
    gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cropped = gray[34:194, :]                    # remove scoreboard → (160,160)
    resized = cv2.resize(cropped, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


class VecFrameStack:
    """Frame-stacker for a batch of N parallel environments."""
    def __init__(self, n_envs, n_stack=FRAME_STACK):
        self.n_envs  = n_envs
        self.n_stack = n_stack
        self.frames  = np.zeros((n_envs, n_stack, FRAME_H, FRAME_W), dtype=np.float32)

    def reset(self, obs_batch):
        """obs_batch: list/array of raw frames, one per env."""
        for i, obs in enumerate(obs_batch):
            proc = preprocess_frame(obs)
            self.frames[i] = np.stack([proc] * self.n_stack, axis=0)
        return self.frames.copy()

    def step(self, obs_batch):
        for i, obs in enumerate(obs_batch):
            proc = preprocess_frame(obs)
            self.frames[i] = np.roll(self.frames[i], shift=-1, axis=0)
            self.frames[i, -1] = proc
        return self.frames.copy()


# ── Actor-Critic Network ───────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    """
    Shared CNN backbone with separate policy (actor) and value (critic) heads.

    Input  : (batch, 4, 84, 84)
    Outputs: action logits (batch, N_ACTIONS)
             state value   (batch, 1)
    """
    def __init__(self, n_actions=N_ACTIONS):
        super().__init__()

        # Shared convolutional feature extractor (Nature DQN architecture)
        self.cnn = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4),  # →(32,20,20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),            # →(64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),            # →(64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),
        )

        cnn_out = self._cnn_out_size()

        self.shared_fc = nn.Sequential(
            nn.Linear(cnn_out, 512),
            nn.ReLU(),
        )

        # Actor head → action logits
        self.actor  = nn.Linear(512, n_actions)

        # Critic head → scalar state value
        self.critic = nn.Linear(512, 1)

        # Orthogonal init (standard for PPO)
        self._init_weights()

    def _cnn_out_size(self):
        dummy = torch.zeros(1, FRAME_STACK, FRAME_H, FRAME_W)
        return int(self.cnn(dummy).shape[1])

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Actor output layer: smaller gain for more uniform initial policy
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x):
        feat    = self.cnn(x)
        shared  = self.shared_fc(feat)
        logits  = self.actor(shared)
        value   = self.critic(shared).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs_np):
        """
        obs_np : numpy (N, 4, 84, 84)
        Returns action indices, log_probs, values (all numpy)
        """
        obs_t           = torch.FloatTensor(obs_np).to(DEVICE)
        logits, values  = self.forward(obs_t)
        probs           = F.softmax(logits, dim=-1)
        dist            = torch.distributions.Categorical(probs)
        actions         = dist.sample()
        log_probs       = dist.log_prob(actions)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    def evaluate(self, obs_t, actions_t):
        """Full forward pass for PPO update — returns log_probs, values, entropy."""
        logits, values = self.forward(obs_t)
        probs          = F.softmax(logits, dim=-1)
        dist           = torch.distributions.Categorical(probs)
        log_probs      = dist.log_prob(actions_t)
        entropy        = dist.entropy()
        return log_probs, values, entropy


# ── GAE Advantage Computation ──────────────────────────────────────────────────
def compute_gae(rewards, values, dones, next_value, gamma=GAMMA, lam=GAE_LAMBDA):
    """
    Generalised Advantage Estimation (GAE-λ):
        δ_t   = r_t + γ·V(s_{t+1})·(1-done) − V(s_t)
        A_t   = δ_t + γλ·A_{t+1}

    Returns advantages and discounted returns (targets for value loss).
    """
    T          = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae   = 0.0

    for t in reversed(range(T)):
        mask       = 1.0 - dones[t]
        next_val   = next_value if t == T - 1 else values[t + 1]
        delta      = rewards[t] + gamma * next_val * mask - values[t]
        last_gae   = delta + gamma * lam * mask * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# ── Rollout Buffer ─────────────────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self, n_envs, rollout_steps):
        T, N = rollout_steps, n_envs
        self.obs       = np.zeros((T, N, FRAME_STACK, FRAME_H, FRAME_W), dtype=np.float32)
        self.actions   = np.zeros((T, N), dtype=np.int64)
        self.log_probs = np.zeros((T, N), dtype=np.float32)
        self.rewards   = np.zeros((T, N), dtype=np.float32)
        self.values    = np.zeros((T, N), dtype=np.float32)
        self.dones     = np.zeros((T, N), dtype=np.float32)
        self.ptr       = 0

    def store(self, obs, actions, log_probs, rewards, values, dones):
        t = self.ptr
        self.obs[t]       = obs
        self.actions[t]   = actions
        self.log_probs[t] = log_probs
        self.rewards[t]   = rewards
        self.values[t]    = values
        self.dones[t]     = dones
        self.ptr         += 1

    def get(self, advantages, returns):
        """Flatten (T×N) → (T*N,) and return shuffled minibatches."""
        T, N = ROLLOUT_STEPS, NUM_ENVS
        obs       = self.obs.reshape(-1, FRAME_STACK, FRAME_H, FRAME_W)
        actions   = self.actions.reshape(-1)
        log_probs = self.log_probs.reshape(-1)
        advs      = advantages.reshape(-1)
        rets      = returns.reshape(-1)

        # Normalise advantages over the entire batch
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        indices = np.random.permutation(T * N)
        for start in range(0, T * N, MINI_BATCH_SIZE):
            idx = indices[start: start + MINI_BATCH_SIZE]
            yield (
                torch.FloatTensor(obs[idx]).to(DEVICE),
                torch.LongTensor(actions[idx]).to(DEVICE),
                torch.FloatTensor(log_probs[idx]).to(DEVICE),
                torch.FloatTensor(advs[idx]).to(DEVICE),
                torch.FloatTensor(rets[idx]).to(DEVICE),
            )

    def reset(self):
        self.ptr = 0


# ── PPO Update ─────────────────────────────────────────────────────────────────
def ppo_update(model, optimizer, buffer, advantages, returns, global_step, total_steps):
    total_pg_loss = total_vf_loss = total_ent = total_clip_frac = 0.0
    n_batches = 0

    for _ in range(PPO_EPOCHS):
        for obs_b, act_b, old_lp_b, adv_b, ret_b in buffer.get(advantages, returns):

            new_lp, values, entropy = model.evaluate(obs_b, act_b)

            # Probability ratio
            ratio = torch.exp(new_lp - old_lp_b)

            # PPO-Clip surrogate objective
            pg_loss1 = -adv_b * ratio
            pg_loss2 = -adv_b * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
            pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss (clipped for stability)
            vf_loss  = F.mse_loss(values, ret_b)

            # Entropy bonus
            ent_loss = -entropy.mean()

            loss = pg_loss + VALUE_COEFF * vf_loss + ENTROPY_COEFF * ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            # LR annealing
            if LR_ANNEAL:
                frac = 1.0 - global_step / total_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = frac * LEARNING_RATE

            clip_frac = ((ratio - 1).abs() > CLIP_EPS).float().mean().item()
            total_pg_loss  += pg_loss.item()
            total_vf_loss  += vf_loss.item()
            total_ent      += (-ent_loss.item())
            total_clip_frac += clip_frac
            n_batches      += 1

    return (total_pg_loss / n_batches,
            total_vf_loss / n_batches,
            total_ent     / n_batches,
            total_clip_frac / n_batches)


# ── Main Training Loop ─────────────────────────────────────────────────────────
def train():
    # Create vectorised environments
    def make_env():
        return gym.make("ALE/Pong-v5")

    envs = [make_env() for _ in range(NUM_ENVS)]

    model     = ActorCritic(N_ACTIONS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-5)
    buffer    = RolloutBuffer(NUM_ENVS, ROLLOUT_STEPS)
    stacker   = VecFrameStack(NUM_ENVS, FRAME_STACK)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"PPO Agent — ALE/Pong-v5")
    print(f"Device     : {DEVICE}")
    print(f"Parameters : {param_count:,}")
    print(f"Envs       : {NUM_ENVS}  |  Rollout steps: {ROLLOUT_STEPS}")
    print(f"Total steps: {TOTAL_TIMESTEPS:,}\n")

    # Reset all envs
    obs_list = [env.reset()[0] for env in envs]
    states   = stacker.reset(obs_list)   # (N, 4, 84, 84)

    global_step    = 0
    episode_count  = 0
    update_count   = 0

    ep_reward_buf  = np.zeros(NUM_ENVS)
    ep_rewards     = []            # completed episode rewards
    ep_steps_buf   = np.zeros(NUM_ENVS)

    log_rewards    = []
    log_pg_losses  = []
    log_vf_losses  = []
    log_entropies  = []
    log_clip_fracs = []

    best_avg       = -float("inf")
    last_ckpt_step = 0
    start_time     = time.time()

    total_updates  = TOTAL_TIMESTEPS // (NUM_ENVS * ROLLOUT_STEPS)

    print(f"{'Update':>7} | {'Steps':>9} | {'Episodes':>8} | "
          f"{'AvgRew':>7} | {'PgLoss':>7} | {'VfLoss':>7} | "
          f"{'Entropy':>8} | {'ClipFrac':>9} | {'SPS':>6}")
    print("─" * 88)

    for update in range(1, total_updates + 1):

        # ── Collect rollout ────────────────────────────────────────────────────
        buffer.reset()

        for step in range(ROLLOUT_STEPS):
            global_step += NUM_ENVS

            actions_idx, log_probs, values = model.act(states)
            # Map reduced action indices → ALE action integers
            env_actions = [VALID_ACTIONS[a] for a in actions_idx]

            next_obs_list, rewards, terminateds, truncateds = [], [], [], []
            for i, env in enumerate(envs):
                obs, rew, term, trunc, _ = env.step(env_actions[i])
                next_obs_list.append(obs)
                rewards.append(rew)
                terminateds.append(term)
                truncateds.append(trunc)

                ep_reward_buf[i] += rew
                ep_steps_buf[i]  += 1

                if term or trunc:
                    ep_rewards.append(ep_reward_buf[i])
                    ep_reward_buf[i] = 0.0
                    ep_steps_buf[i]  = 0.0
                    episode_count   += 1
                    reset_obs, _ = env.reset()
                    next_obs_list[i] = reset_obs

            dones      = np.array([t or tr for t, tr in zip(terminateds, truncateds)], dtype=np.float32)
            rewards_np = np.array(rewards, dtype=np.float32)

            next_states = stacker.step(next_obs_list)

            buffer.store(states, actions_idx, log_probs, rewards_np, values, dones)
            states = next_states

        # Bootstrap value for last state
        with torch.no_grad():
            _, next_values = model.forward(
                torch.FloatTensor(states).to(DEVICE)
            )
            next_value_np = next_values.cpu().numpy()

        # Compute GAE advantages per env, then stack
        all_advantages = np.zeros((ROLLOUT_STEPS, NUM_ENVS), dtype=np.float32)
        all_returns    = np.zeros((ROLLOUT_STEPS, NUM_ENVS), dtype=np.float32)

        for env_i in range(NUM_ENVS):
            adv, ret = compute_gae(
                buffer.rewards[:, env_i],
                buffer.values[:, env_i],
                buffer.dones[:, env_i],
                next_value_np[env_i],
            )
            all_advantages[:, env_i] = adv
            all_returns[:, env_i]    = ret

        # ── PPO update ─────────────────────────────────────────────────────────
        pg_loss, vf_loss, entropy, clip_frac = ppo_update(
            model, optimizer, buffer, all_advantages, all_returns,
            global_step, TOTAL_TIMESTEPS
        )
        update_count += 1

        # ── Logging ────────────────────────────────────────────────────────────
        sps       = int(global_step / (time.time() - start_time))
        avg_rew   = np.mean(ep_rewards[-50:]) if ep_rewards else float("nan")

        log_rewards.append(avg_rew)
        log_pg_losses.append(pg_loss)
        log_vf_losses.append(vf_loss)
        log_entropies.append(entropy)
        log_clip_fracs.append(clip_frac)

        if update % 10 == 0:
            print(
                f"{update:>7} | {global_step:>9,} | {episode_count:>8} | "
                f"{avg_rew:>7.2f} | {pg_loss:>7.4f} | {vf_loss:>7.4f} | "
                f"{entropy:>8.4f} | {clip_frac:>9.4f} | {sps:>6}"
            )

        # Save best model
        if len(ep_rewards) >= 50 and avg_rew > best_avg:
            best_avg = avg_rew
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "global_step": global_step,
                "best_avg": best_avg,
            }, POLICY_PATH)
            print(f"  ★ New best avg-50: {best_avg:.2f}  → saved '{POLICY_PATH}'")

        # Periodic checkpoints
        if global_step - last_ckpt_step >= SAVE_EVERY_STEPS:
            ckpt = f"{CHECKPOINT_PREFIX}_{global_step:07d}.pth"
            torch.save({
                "model_state": model.state_dict(),
                "global_step": global_step,
            }, ckpt)
            last_ckpt_step = global_step
            print(f"  ✔ Checkpoint: '{ckpt}'")

    for env in envs:
        env.close()

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} min.")
    print(f"Best avg-50 reward: {best_avg:.2f}")

    # ── Training curve ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("ALE/Pong-v5 — PPO Training Diagnostics", fontsize=13, fontweight="bold")

    x = np.arange(len(log_rewards))

    axes[0, 0].plot(x, log_rewards,    color="steelblue",  linewidth=1.5)
    axes[0, 0].axhline(0,  color="gray",  linestyle="--", linewidth=0.8)
    axes[0, 0].axhline(21, color="green", linestyle="--", linewidth=0.8, label="Max +21")
    axes[0, 0].set_title("Avg Reward (last 50 eps)")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(x, log_pg_losses,  color="tomato",    linewidth=1.2)
    axes[0, 1].set_title("Policy Loss")
    axes[0, 1].set_ylabel("Loss")

    axes[1, 0].plot(x, log_vf_losses,  color="seagreen",  linewidth=1.2)
    axes[1, 0].set_title("Value Loss")
    axes[1, 0].set_ylabel("MSE Loss")
    axes[1, 0].set_xlabel("Update")

    axes[1, 1].plot(x, log_entropies,  color="darkorange", linewidth=1.2, label="Entropy")
    ax2 = axes[1, 1].twinx()
    ax2.plot(x, log_clip_fracs, color="purple", linewidth=1.0, linestyle="--", label="Clip frac")
    axes[1, 1].set_title("Entropy & Clip Fraction")
    axes[1, 1].set_ylabel("Entropy")
    ax2.set_ylabel("Clip Fraction")
    axes[1, 1].set_xlabel("Update")

    # Combined legend
    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    plt.tight_layout()
    plt.savefig("pong_ppo_training_curve.png", dpi=150)
    plt.show()
    print("Training curve saved → 'pong_ppo_training_curve.png'")


if __name__ == "__main__":
    train()