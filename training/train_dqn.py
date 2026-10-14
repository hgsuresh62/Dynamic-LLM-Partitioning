
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from training.rl_environment import PartitioningEnvironment
from training.dqn_agent import DQN
from training.replay_buffer import ReplayBuffer


# ============================================================
# CONFIGURATION
# ============================================================

NUM_EPISODES = 50
MAX_STEPS_PER_EPISODE = 10

BATCH_SIZE = 8
REPLAY_CAPACITY = 5000

GAMMA = 0.99
LEARNING_RATE = 0.001

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.97

TARGET_UPDATE_FREQUENCY = 10
SEED = 42

STATE_SIZE = 5
ACTION_SIZE = 3

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "dqn_policy_5features.pt",
)

# Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")


# ============================================================
# INITIALIZATION
# ============================================================

print("=" * 65)
print("FIVE-FEATURE DQN TRAINING")
print("=" * 65)
print(f"Device: {device}")
print(f"Episodes: {NUM_EPISODES}")
print(f"Steps per episode: {MAX_STEPS_PER_EPISODE}")
print(f"State features: {STATE_SIZE}")
print(f"Actions: {ACTION_SIZE}")
print(f"Checkpoint: {MODEL_PATH}")
print("=" * 65)

env = PartitioningEnvironment()

policy_net = DQN(
    state_size=STATE_SIZE,
    action_size=ACTION_SIZE,
).to(device)

target_net = DQN(
    state_size=STATE_SIZE,
    action_size=ACTION_SIZE,
).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(
    policy_net.parameters(),
    lr=LEARNING_RATE,
)

loss_function = nn.SmoothL1Loss()

replay_buffer = ReplayBuffer(
    capacity=REPLAY_CAPACITY,
)


# ============================================================
# ACTION SELECTION
# ============================================================

def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randrange(ACTION_SIZE)

    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = policy_net(state_tensor)

    return int(torch.argmax(q_values, dim=1).item())


# ============================================================
# TRAINING STEP
# ============================================================

def train_step():
    if not replay_buffer.can_sample(BATCH_SIZE):
        return None

    batch = replay_buffer.sample(BATCH_SIZE)

    states = np.asarray(
        [item[0] for item in batch],
        dtype=np.float32,
    )
    actions = np.asarray(
        [item[1] for item in batch],
        dtype=np.int64,
    )
    rewards = np.asarray(
        [item[2] for item in batch],
        dtype=np.float32,
    )
    next_states = np.asarray(
        [item[3] for item in batch],
        dtype=np.float32,
    )
    dones = np.asarray(
        [item[4] for item in batch],
        dtype=np.float32,
    )

    # Fail early if any state has the wrong dimensions.
    if states.shape[1] != STATE_SIZE:
        raise ValueError(
            f"Expected {STATE_SIZE} state features; "
            f"received {states.shape[1]}."
        )

    if next_states.shape[1] != STATE_SIZE:
        raise ValueError(
            f"Expected {STATE_SIZE} next-state features; "
            f"received {next_states.shape[1]}."
        )

    states_tensor = torch.as_tensor(
        states, dtype=torch.float32, device=device
    )
    actions_tensor = torch.as_tensor(
        actions, dtype=torch.long, device=device
    ).unsqueeze(1)
    rewards_tensor = torch.as_tensor(
        rewards, dtype=torch.float32, device=device
    )
    next_states_tensor = torch.as_tensor(
        next_states, dtype=torch.float32, device=device
    )
    dones_tensor = torch.as_tensor(
        dones, dtype=torch.float32, device=device
    )

    # Q(s, a) from the online network.
    current_q_values = policy_net(
        states_tensor
    ).gather(
        1, actions_tensor
    ).squeeze(1)

    # Bellman target from the target network.
    with torch.no_grad():
        next_q_values = target_net(next_states_tensor)
        max_next_q_values = next_q_values.max(dim=1).values

        target_q_values = rewards_tensor + (
            GAMMA
            * max_next_q_values
            * (1.0 - dones_tensor)
        )

    loss = loss_function(
        current_q_values,
        target_q_values,
    )

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        policy_net.parameters(),
        max_norm=1.0,
    )

    optimizer.step()

    return float(loss.item())


# ============================================================
# TRAINING LOOP
# ============================================================

epsilon = EPSILON_START

episode_rewards = []
episode_losses = []

for episode in range(1, NUM_EPISODES + 1):
    state = env.reset()

    if len(state) != STATE_SIZE:
        raise ValueError(
            f"Environment returned {len(state)} features; "
            f"expected {STATE_SIZE}."
        )

    total_reward = 0.0
    losses = []

    for step in range(MAX_STEPS_PER_EPISODE):
        action = select_action(state, epsilon)

        next_state, reward, done, info = env.step(action)

        if len(next_state) != STATE_SIZE:
            raise ValueError(
                f"Environment returned {len(next_state)} "
                f"next-state features; expected {STATE_SIZE}."
            )

        replay_buffer.push(
            state,
            action,
            reward,
            next_state,
            done,
        )

        loss = train_step()

        if loss is not None:
            losses.append(loss)

        state = next_state
        total_reward += reward

        if done:
            break

    # Periodically synchronize the target network.
    if episode % TARGET_UPDATE_FREQUENCY == 0:
        target_net.load_state_dict(policy_net.state_dict())

    epsilon = max(
        EPSILON_END,
        epsilon * EPSILON_DECAY,
    )

    episode_rewards.append(total_reward)

    average_loss = (
        float(np.mean(losses)) if losses else 0.0
    )
    episode_losses.append(average_loss)

    print(
        f"Episode {episode:02d}/{NUM_EPISODES} | "
        f"Reward: {total_reward:8.4f} | "
        f"Loss: {average_loss:8.5f} | "
        f"Epsilon: {epsilon:.4f} | "
        f"Replay: {len(replay_buffer)}"
    )


# ============================================================
# SAVE FIVE-FEATURE CHECKPOINT
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)

torch.save(
    {
        "model_state_dict": policy_net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "state_size": STATE_SIZE,
        "action_size": ACTION_SIZE,
        "episodes": NUM_EPISODES,
        "epsilon": epsilon,
        "episode_rewards": episode_rewards,
        "episode_losses": episode_losses,
    },
    MODEL_PATH,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 65)
print("FIVE-FEATURE DQN TRAINING COMPLETE")
print("=" * 65)

print(f"Final epsilon: {epsilon:.4f}")
print(f"Experiences collected: {len(replay_buffer)}")
print(f"Average reward: {np.mean(episode_rewards):.4f}")
print(
    "Last 10 episode average reward: "
    f"{np.mean(episode_rewards[-10:]):.4f}"
)
print(f"Final average loss: {episode_losses[-1]:.5f}")
print(f"Model saved to: {MODEL_PATH}")
print("=" * 65)
