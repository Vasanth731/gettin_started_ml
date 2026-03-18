"""
train.py  –  DQN training for ALE/Pacman-v5
============================================
Observation : RGB (210, 160, 3)  →  grayscale + resize + frame-stack (4)
Actions     : 5  (NOOP, UP, RIGHT, LEFT, DOWN)
Algorithm   : DQN with experience replay + target network

Install deps:
    pip install gymnasium[atari] ale-py torch numpy opencv-python
"""

# # pip install gymnasium[atari] ale-py
# # pip install autorom[accept-rom-license] autorom --accept-license


import ale_py
import gymnasium as gym
gym.register_envs(ale_py)
import random
import collections
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# ── Hyperparameters ───────────────────────────────────────────────────────────
ENV_ID          = "ALE/Pacman-v5"
SAVE_PATH       = "pacman_best.pth"

EPISODES        = 5000
MAX_STEPS       = 10_000        # max steps per episode

# CNN input
FRAME_H         = 84
FRAME_W         = 84
FRAME_STACK     = 4             # stack last 4 frames for motion info

# DQN
GAMMA           = 0.99
LR              = 1e-4
BATCH_SIZE      = 32
REPLAY_SIZE     = 50_000        # experience replay buffer size
REPLAY_MIN      = 10_000        # start training only after this many samples
TARGET_UPDATE   = 1000          # steps between target network syncs

# Exploration (epsilon-greedy)
EPS_START       = 1.0
EPS_END         = 0.05
EPS_DECAY       = 200_000       # steps to decay epsilon over

PRINT_EVERY     = 20


# ── Frame preprocessing ───────────────────────────────────────────────────────
def preprocess(frame: np.ndarray) -> np.ndarray:
    """RGB (210,160,3) → grayscale (84,84) uint8"""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
    return resized                          # shape (84, 84)


# ── Frame stack wrapper ───────────────────────────────────────────────────────
class FrameStack:
    """Maintains a rolling buffer of the last FRAME_STACK preprocessed frames."""
    def __init__(self, n: int = FRAME_STACK):
        self.n = n
        self.frames = collections.deque(maxlen=n)

    def reset(self, frame):
        f = preprocess(frame)
        for _ in range(self.n):
            self.frames.append(f)
        return self._get()

    def step(self, frame):
        self.frames.append(preprocess(frame))
        return self._get()

    def _get(self) -> np.ndarray:
        # shape (4, 84, 84)  float32 in [0, 1]
        return np.stack(self.frames, axis=0).astype(np.float32) / 255.0


# ── CNN Policy (DQN) ──────────────────────────────────────────────────────────
class DQN(nn.Module):
    """
    Classic Atari DQN architecture (Mnih et al. 2015).
    Input : (batch, 4, 84, 84)
    Output: (batch, n_actions)  Q-values
    """
    def __init__(self, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4),  # → (32,20,20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),           # → (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),           # → (64, 7, 7)
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


# ── Experience Replay Buffer ──────────────────────────────────────────────────
Transition = collections.namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buf)


# ── Epsilon schedule ──────────────────────────────────────────────────────────
def get_epsilon(step: int) -> float:
    return EPS_END + (EPS_START - EPS_END) * max(0, 1 - step / EPS_DECAY)


# ── DQN update ────────────────────────────────────────────────────────────────
def update_dqn(
    policy_net: DQN,
    target_net: DQN,
    optimizer:  optim.Optimizer,
    buffer:     ReplayBuffer,
    device:     torch.device,
) -> float:
    batch = buffer.sample(BATCH_SIZE)

    states      = torch.tensor(np.array(batch.state),      dtype=torch.float32).to(device)
    actions     = torch.tensor(batch.action,               dtype=torch.long).to(device)
    rewards     = torch.tensor(batch.reward,               dtype=torch.float32).to(device)
    next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(device)
    dones       = torch.tensor(batch.done,                 dtype=torch.float32).to(device)

    # Current Q values for taken actions
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Target Q values (Bellman)
    with torch.no_grad():
        max_next_q = target_net(next_states).max(1)[0]
        targets    = rewards + GAMMA * max_next_q * (1 - dones)

    loss = nn.functional.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()
    return loss.item()


# ── Training loop ─────────────────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    env    = gym.make(ENV_ID)
    stacker = FrameStack()
    n_actions = env.action_space.n          # 5

    policy_net = DQN(n_actions).to(device)
    target_net = DQN(n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    buffer    = ReplayBuffer(REPLAY_SIZE)

    best_mean_reward = -float("inf")
    recent_rewards: list[float] = []
    total_steps = 0

    print(f"Training {ENV_ID}  |  actions={n_actions}")
    print(f"Weights → {SAVE_PATH}\n")

    for episode in range(1, EPISODES + 1):
        obs, _ = env.reset()
        state  = stacker.reset(obs)
        ep_reward = 0.0

        for _ in range(MAX_STEPS):
            # ── Epsilon-greedy action ─────────────────────────────────────────
            eps = get_epsilon(total_steps)
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    action  = policy_net(state_t).argmax(1).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            next_state = stacker.step(obs)
            done = terminated or truncated

            # Clip reward to [-1, 1] for training stability
            clipped_reward = np.clip(reward, -1, 1)
            buffer.push(state, action, clipped_reward, next_state, float(done))

            state      = next_state
            ep_reward += reward
            total_steps += 1

            # ── Train ─────────────────────────────────────────────────────────
            if len(buffer) >= REPLAY_MIN:
                update_dqn(policy_net, target_net, optimizer, buffer, device)

            # ── Sync target network ───────────────────────────────────────────
            if total_steps % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        recent_rewards.append(ep_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
        mean_reward = float(np.mean(recent_rewards))

        # ── Save best weights ─────────────────────────────────────────────────
        if mean_reward > best_mean_reward and len(recent_rewards) == 100:
            best_mean_reward = mean_reward
            torch.save(
                {
                    "episode":          episode,
                    "model_state_dict": policy_net.state_dict(),
                    "best_mean_reward": best_mean_reward,
                    "n_actions":        n_actions,
                },
                SAVE_PATH,
            )
            print(f"  ✔ New best saved  ep={episode}  mean-100={best_mean_reward:.1f}")

        if episode % PRINT_EVERY == 0:
            print(
                f"Ep {episode:5d}/{EPISODES}  "
                f"reward={ep_reward:8.1f}  "
                f"mean-100={mean_reward:8.1f}  "
                f"best={best_mean_reward:8.1f}  "
                f"eps={get_epsilon(total_steps):.3f}  "
                f"buf={len(buffer)}"
            )

    env.close()
    print(f"\nTraining complete. Best weights → {SAVE_PATH}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()