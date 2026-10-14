import random


# ============================================================
# RL ENVIRONMENT
# Dynamic RL-Based LLM Execution Graph Partitioning
# ============================================================


class PartitioningEnvironment:

    def __init__(self):

        # ----------------------------------------------------
        # Model configuration
        # ----------------------------------------------------

        self.total_layers = 28

        # Start from the best static split discovered
        # during Week 1.
        self.split_point = 12

        # Allowed split range
        self.min_split = 8
        self.max_split = 20

        # ----------------------------------------------------
        # RL actions
        # ----------------------------------------------------

        self.actions = {
            0: "Keep Split",
            1: "Shift +1 Layer Left",
            2: "Shift +1 Layer Right"
        }

        self.action_size = 3

        # ----------------------------------------------------
        # Environment state
        # ----------------------------------------------------

        self.current_token = 1
        self.total_tokens = 10

        self.state = None


    # ========================================================
    # RESET ENVIRONMENT
    # ========================================================

    def reset(self):

        self.split_point = 12

        self.current_token = 1

        self.state = self._generate_state()

        return self.state


    # ========================================================
    # GENERATE SIMULATED TELEMETRY STATE
    # ========================================================

    def _generate_state(self):

        # Simulated worker utilization
        worker_1_utilization = random.uniform(
            0.40,
            0.80
        )

        worker_2_utilization = random.uniform(
            0.40,
            0.80
        )

        # Simulated network latency
        network_latency = random.uniform(
            0.02,
            1.00
        )

        # Token sequence position
        token_position = (
            self.current_token
            / self.total_tokens
        )

        state = [

            worker_1_utilization,

            worker_2_utilization,

            network_latency,

            token_position
        ]

        return state


    # ========================================================
    # SIMULATE EXECUTION TIME
    # ========================================================

    def _simulate_execution_time(self):

        # Base computation time.
        #
        # The model has 28 layers. A perfectly balanced
        # split is not necessarily the fastest, which is
        # why we include a small imbalance component.

        worker_1_layers = self.split_point

        worker_2_layers = (
            self.total_layers
            - self.split_point
        )

        # Approximate computation cost
        worker_1_time = (
            worker_1_layers * 0.050
        )

        worker_2_time = (
            worker_2_layers * 0.050
        )

        # Simulated network latency
        network_latency = random.uniform(
            0.002,
            0.100
        )

        # Small random runtime variation
        runtime_noise = random.uniform(
            0.000,
            0.020
        )

        total_time = (
            max(
                worker_1_time,
                worker_2_time
            )
            + network_latency
            + runtime_noise
        )

        return total_time


    # ========================================================
    # CALCULATE REWARD
    # ========================================================

    def _calculate_reward(
        self,
        execution_time,
        previous_time,
        action
    ):

        # Faster execution = positive reward
        speed_reward = (
            previous_time
            - execution_time
        )

        # ----------------------------------------------------
        # Layer movement penalty
        # ----------------------------------------------------

        movement_penalty = 0.0

        if action != 0:

            movement_penalty = 0.005

        # ----------------------------------------------------
        # Final reward
        # ----------------------------------------------------

        reward = (
            speed_reward
            - movement_penalty
        )

        return reward


    # ========================================================
    # STEP
    # ========================================================

    def step(self, action):

        if action not in self.actions:

            raise ValueError(
                f"Invalid action: {action}"
            )

        # ----------------------------------------------------
        # Previous performance
        # ----------------------------------------------------

        previous_time = (
            self._simulate_execution_time()
        )

        previous_split = self.split_point

        # ----------------------------------------------------
        # Apply action
        # ----------------------------------------------------

        if action == 0:

            # Keep current split
            pass

        elif action == 1:

            # Move boundary one layer left
            if self.split_point > self.min_split:

                self.split_point -= 1

        elif action == 2:

            # Move boundary one layer right
            if self.split_point < self.max_split:

                self.split_point += 1

        # ----------------------------------------------------
        # New performance
        # ----------------------------------------------------

        execution_time = (
            self._simulate_execution_time()
        )

        # ----------------------------------------------------
        # Reward
        # ----------------------------------------------------

        reward = self._calculate_reward(
            execution_time,
            previous_time,
            action
        )

        # ----------------------------------------------------
        # Advance token
        # ----------------------------------------------------

        self.current_token += 1

        done = (
            self.current_token
            > self.total_tokens
        )

        if not done:

            self.state = self._generate_state()

        else:

            self.state = self._generate_state()

        # ----------------------------------------------------
        # Additional information
        # ----------------------------------------------------

        info = {

            "previous_split": previous_split,

            "new_split": self.split_point,

            "action_name":
                self.actions[action],

            "previous_time":
                previous_time,

            "execution_time":
                execution_time
        }

        return (
            self.state,
            reward,
            done,
            info
        )


# ============================================================
# TEST ENVIRONMENT
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("RL PARTITIONING ENVIRONMENT TEST")
    print("=" * 70)

    environment = PartitioningEnvironment()

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    state = environment.reset()

    print("\nEnvironment reset successfully.")

    print(
        f"Initial split: "
        f"{environment.split_point} / "
        f"{environment.total_layers - environment.split_point}"
    )

    print(
        f"Initial state: "
        f"{state}"
    )

    print(
        f"State size: "
        f"{len(state)}"
    )

    print(
        f"Number of actions: "
        f"{environment.action_size}"
    )


    # --------------------------------------------------------
    # Test all actions
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("TESTING RL ACTIONS")
    print("-" * 70)

    for action in range(
        environment.action_size
    ):

        # Reset before each action
        environment.reset()

        old_split = environment.split_point

        (
            new_state,
            reward,
            done,
            info
        ) = environment.step(action)

        print(
            f"\nAction {action}: "
            f"{environment.actions[action]}"
        )

        print(
            f"Previous split: "
            f"{old_split} / "
            f"{28 - old_split}"
        )

        print(
            f"New split: "
            f"{info['new_split']} / "
            f"{28 - info['new_split']}"
        )

        print(
            f"Previous execution time: "
            f"{info['previous_time']:.4f}s"
        )

        print(
            f"New execution time: "
            f"{info['execution_time']:.4f}s"
        )

        print(
            f"Reward: "
            f"{reward:.6f}"
        )

        print(
            f"New state: "
            f"{new_state}"
        )


    # ========================================================
    # VALIDATION
    # ========================================================

    print("\n" + "-" * 70)
    print("ENVIRONMENT VALIDATION")
    print("-" * 70)

    environment.reset()

    if len(environment.state) == 4:

        print(
            "PASS: State contains 4 features."
        )

    else:

        print(
            "FAIL: Incorrect state size."
        )


    if environment.action_size == 3:

        print(
            "PASS: Environment contains 3 actions."
        )

    else:

        print(
            "FAIL: Incorrect action count."
        )


    # Test left movement
    environment.split_point = 12

    environment.step(1)

    if environment.split_point == 11:

        print(
            "PASS: Left action shifts split correctly."
        )

    else:

        print(
            "FAIL: Left action failed."
        )


    # Test right movement
    environment.split_point = 12

    environment.step(2)

    if environment.split_point == 13:

        print(
            "PASS: Right action shifts split correctly."
        )

    else:

        print(
            "FAIL: Right action failed."
        )


    # Test keep
    environment.split_point = 12

    environment.step(0)

    if environment.split_point == 12:

        print(
            "PASS: Keep action preserves split."
        )

    else:

        print(
            "FAIL: Keep action changed split."
        )


    print("\n" + "=" * 70)
    print("RL ENVIRONMENT TEST COMPLETED")
    print("=" * 70)