
import torch
import torch.nn as nn


# ============================================================
# DQN NETWORK
# Dynamic RL-Based LLM Execution Graph Partitioning
# ============================================================


class DQN(nn.Module):
    def __init__(self, state_size=5, action_size=3):
        super().__init__()

        # Five input features:
        # 1. Worker 1 utilization
        # 2. Worker 2 utilization
        # 3. Network latency
        # 4. Token position
        # 5. Current split point / total layers
        #
        # Three output actions:
        # 0 = Keep Split
        # 1 = Shift +1 Layer Left
        # 2 = Shift +1 Layer Right

        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, state):
        return self.network(state)


# ============================================================
# TEST DQN
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DQN NETWORK TEST")
    print("=" * 70)

    STATE_SIZE = 5
    ACTION_SIZE = 3

    print(f"\nInput state size: {STATE_SIZE}")
    print(f"Number of actions: {ACTION_SIZE}")

    dqn = DQN(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
    )

    print("\nDQN architecture:")
    print(dqn)

    total_parameters = sum(
        parameter.numel()
        for parameter in dqn.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in dqn.parameters()
        if parameter.requires_grad
    )

    print(f"\nTotal parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")

    # Sample five-feature state.
    sample_state = torch.tensor(
        [[0.70, 0.50, 0.40, 0.50, 12 / 28]],
        dtype=torch.float32,
    )

    print("\nSample state:")
    print(sample_state)

    # Forward pass.
    with torch.no_grad():
        q_values = dqn(sample_state)

    print("\nQ-values:")
    print(q_values)

    print(f"\nQ-value shape: {tuple(q_values.shape)}")

    selected_action = torch.argmax(
        q_values,
        dim=1,
    ).item()

    action_names = {
        0: "Keep Split",
        1: "Shift +1 Layer Left",
        2: "Shift +1 Layer Right",
    }

    print("\nSelected action:")
    print(
        f"Action {selected_action}: "
        f"{action_names[selected_action]}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\n" + "-" * 70)
    print("DQN VALIDATION")
    print("-" * 70)

    assert q_values.shape == (1, ACTION_SIZE)
    print("PASS: DQN produces exactly 3 Q-values.")

    assert torch.isfinite(q_values).all()
    print("PASS: All Q-values are finite.")

    assert selected_action in action_names
    print("PASS: DQN selects a valid action.")

    assert sample_state.shape == (1, STATE_SIZE)
    print("PASS: DQN accepts five input features.")

    print("\n" + "=" * 70)
    print("DQN NETWORK TEST COMPLETED")
    print("=" * 70)
