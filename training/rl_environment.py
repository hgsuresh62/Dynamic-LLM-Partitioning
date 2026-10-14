
import random


class PartitioningEnvironment:
    """
    Simulated RL environment for LLM layer partitioning.

    State:
        0: Worker 1 utilization
        1: Worker 2 utilization
        2: Network latency
        3: Token position
        4: Current split point / total layers

    Actions:
        0 = Keep split
        1 = Shift one layer left
        2 = Shift one layer right

    Execution times are simulated estimates, not measurements
    of real Llama inference.
    """

    def __init__(self):
        self.total_layers = 28
        self.min_split = 8
        self.max_split = 20
        self.action_size = 3

        self.actions = {
            0: "Keep Split",
            1: "Shift +1 Layer Left",
            2: "Shift +1 Layer Right",
        }

        self.split_point = 12
        self.current_token = 1
        self.total_tokens = 10
        self.state = None

    def reset(self):
        """Reset the environment for a new episode."""
        self.split_point = 12
        self.current_token = 1
        self.state = self._generate_state()
        return self.state

    def _generate_state(self):
        """Generate telemetry and encode the current split."""
        return [
            random.uniform(0.40, 0.80),
            random.uniform(0.40, 0.80),
            random.uniform(0.02, 1.00),
            self.current_token / self.total_tokens,
            self.split_point / self.total_layers,
        ]

    def _estimate_execution_time(self, split_point, state):
        """
        Estimate partition cost using worker utilization,
        layer counts, and communication latency.
        """
        worker_1_util = max(0.0, min(1.0, state[0]))
        worker_2_util = max(0.0, min(1.0, state[1]))
        network_latency = max(0.0, state[2])

        worker_1_layers = split_point
        worker_2_layers = self.total_layers - split_point

        worker_1_time = (
            worker_1_layers
            * 0.050
            * (1.0 + worker_1_util)
        )

        worker_2_time = (
            worker_2_layers
            * 0.050
            * (1.0 + worker_2_util)
        )

        return max(worker_1_time, worker_2_time) + network_latency

    def _calculate_reward(
        self, previous_time, execution_time, action
    ):
        """Reward lower estimated cost and penalize movement."""
        speed_reward = previous_time - execution_time
        movement_penalty = 0.005 if action != 0 else 0.0
        return speed_reward - movement_penalty

    def step(self, action):
        """Apply an action and return the next transition."""
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}")

        if self.state is None:
            raise RuntimeError(
                "Call reset() before calling step()."
            )

        previous_split = self.split_point
        state_used = list(self.state)

        previous_time = self._estimate_execution_time(
            previous_split, state_used
        )

        if action == 1 and self.split_point > self.min_split:
            self.split_point -= 1
        elif action == 2 and self.split_point < self.max_split:
            self.split_point += 1

        execution_time = self._estimate_execution_time(
            self.split_point, state_used
        )

        reward = self._calculate_reward(
            previous_time, execution_time, action
        )

        self.current_token += 1
        done = self.current_token > self.total_tokens

        # The next state reflects the updated split point.
        self.state = self._generate_state()

        info = {
            "previous_split": previous_split,
            "new_split": self.split_point,
            "action_name": self.actions[action],
            "previous_time": previous_time,
            "execution_time": execution_time,
            "state_used": state_used,
        }

        return self.state, reward, done, info


if __name__ == "__main__":
    print("=" * 65)
    print("RL ENVIRONMENT VALIDATION")
    print("=" * 65)

    env = PartitioningEnvironment()
    state = env.reset()

    print("Initial state:", state)
    print("State size:", len(state))
    print(
        "Initial split:",
        f"{env.split_point}/{env.total_layers - env.split_point}",
    )

    assert len(state) == 5
    assert env.action_size == 3
    assert abs(state[4] - 12 / 28) < 1e-9

    # All actions start with the same telemetry and split.
    fixed_state = [0.60, 0.60, 0.05, 0.50, 12 / 28]

    for action, expected_split in [
        (0, 12),
        (1, 11),
        (2, 13),
    ]:
        env.reset()
        env.state = fixed_state.copy()

        _, reward, _, info = env.step(action)

        assert env.split_point == expected_split

        print(
            f"Action {action}: "
            f"{info['previous_split']} -> {info['new_split']}, "
            f"estimated time={info['execution_time']:.4f}s, "
            f"reward={reward:.4f}"
        )

    # Deterministic cost for identical inputs.
    time_a = env._estimate_execution_time(12, fixed_state)
    time_b = env._estimate_execution_time(12, fixed_state)
    assert time_a == time_b

    # Higher utilization increases estimated worker cost.
    low_util_state = [0.20, 0.20, 0.05, 0.50, 12 / 28]
    high_util_state = [0.80, 0.80, 0.05, 0.50, 12 / 28]

    low_util_time = env._estimate_execution_time(
        12, low_util_state
    )
    high_util_time = env._estimate_execution_time(
        12, high_util_state
    )

    assert high_util_time > low_util_time

    # Split boundaries must be respected.
    env.reset()
    env.split_point = env.max_split
    env.state = [
        0.60, 0.60, 0.05, 0.50,
        env.split_point / env.total_layers,
    ]
    env.step(2)
    assert env.split_point == env.max_split

    env.reset()
    env.split_point = env.min_split
    env.state = [
        0.60, 0.60, 0.05, 0.50,
        env.split_point / env.total_layers,
    ]
    env.step(1)
    assert env.split_point == env.min_split

    print()
    print("PASS: Five-feature state preserved.")
    print("PASS: Split point is encoded correctly.")
    print("PASS: All three actions validated.")
    print("PASS: Cost estimate is deterministic.")
    print("PASS: Higher utilization increases estimated cost.")
    print("PASS: Split boundaries enforced.")
    print("NOTE: This validates a simulation, not real inference.")
    print("=" * 65)
