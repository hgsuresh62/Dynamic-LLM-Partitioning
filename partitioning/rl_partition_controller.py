import os
import torch
from training.dqn_agent import DQN


# ============================================================
# Configuration
# ============================================================

STATE_SIZE = 4
ACTION_SIZE = 3

MIN_SPLIT = 8
MAX_SPLIT = 20

MODEL_PATH = os.path.join(
    "models",
    "dqn_policy.pt"
)

device = torch.device("cpu")


# ============================================================
# Action Definitions
# ============================================================

ACTION_NAMES = {
    0: "Keep Split",
    1: "Shift +1 Layer Left",
    2: "Shift +1 Layer Right"
}


# ============================================================
# RL Partition Controller
# ============================================================

class RLPartitionController:

    def __init__(self, model_path=MODEL_PATH):

        self.model = DQN(
            state_size=STATE_SIZE,
            action_size=ACTION_SIZE
        ).to(device)

        checkpoint = torch.load(
            model_path,
            map_location=device
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        self.current_split = 12

    # --------------------------------------------------------
    # Select action using trained DQN
    # --------------------------------------------------------

    def select_action(self, state):

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.model(
                state_tensor
            )

        action = int(
            torch.argmax(
                q_values,
                dim=1
            ).item()
        )

        return action, q_values.squeeze(0)

    # --------------------------------------------------------
    # Convert action into new split point
    # --------------------------------------------------------

    def update_split(self, action):

        old_split = self.current_split

        if action == 0:
            new_split = old_split

        elif action == 1:
            new_split = old_split - 1

        elif action == 2:
            new_split = old_split + 1

        else:
            raise ValueError(
                f"Invalid action: {action}"
            )

        # Keep split within allowed range
        new_split = max(
            MIN_SPLIT,
            min(MAX_SPLIT, new_split)
        )

        self.current_split = new_split

        return old_split, new_split

    # --------------------------------------------------------
    # Complete decision
    # --------------------------------------------------------

    def decide(self, state):

        action, q_values = self.select_action(
            state
        )

        old_split, new_split = self.update_split(
            action
        )

        return {
            "state": state,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "q_values": q_values.tolist(),
            "old_split": old_split,
            "new_split": new_split
        }


# ============================================================
# Test Controller
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("RL PARTITION CONTROLLER")
    print("=" * 60)

    controller = RLPartitionController()

    print(
        f"Initial split: "
        f"{controller.current_split}/"
        f"{28 - controller.current_split}"
    )

    test_states = [
        [0.50, 0.50, 0.02, 0.10],
        [0.80, 0.40, 0.10, 0.30],
        [0.40, 0.80, 0.20, 0.50],
        [0.70, 0.70, 0.50, 0.70],
        [0.30, 0.60, 0.90, 0.90]
    ]

    print()
    print("-" * 60)
    print("DYNAMIC SPLIT DECISIONS")
    print("-" * 60)

    for index, state in enumerate(
        test_states,
        start=1
    ):

        result = controller.decide(state)

        print()
        print(f"Test {index}")
        print(f"State: {result['state']}")

        print(
            f"Q-values: "
            f"{[round(q, 6) for q in result['q_values']]}"
        )

        print(
            f"Action: "
            f"{result['action']} - "
            f"{result['action_name']}"
        )

        print(
            f"Split: "
            f"{result['old_split']}/"
            f"{28 - result['old_split']} "
            f"-> "
            f"{result['new_split']}/"
            f"{28 - result['new_split']}"
        )

    print()
    print("=" * 60)
    print("VALIDATION")
    print("=" * 60)

    assert MIN_SPLIT <= controller.current_split <= MAX_SPLIT

    print("PASS: RL controller produced a valid split point.")
    print("=" * 60)