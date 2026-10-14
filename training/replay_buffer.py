import random
from collections import deque


# ============================================================
# DQN REPLAY BUFFER
# Dynamic RL-Based LLM Execution Graph Partitioning
# ============================================================


class ReplayBuffer:

    def __init__(self, capacity=10000):

        self.buffer = deque(
            maxlen=capacity
        )

        self.capacity = capacity


    # ========================================================
    # STORE EXPERIENCE
    # ========================================================

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        experience = (
            state,
            action,
            reward,
            next_state,
            done
        )

        self.buffer.append(
            experience
        )


    # ========================================================
    # SAMPLE RANDOM BATCH
    # ========================================================

    def sample(self, batch_size):

        if batch_size > len(self.buffer):

            raise ValueError(
                f"Not enough experiences. "
                f"Requested {batch_size}, "
                f"but buffer contains {len(self.buffer)}."
            )

        return random.sample(
            self.buffer,
            batch_size
        )


    # ========================================================
    # BUFFER SIZE
    # ========================================================

    def __len__(self):

        return len(self.buffer)


    # ========================================================
    # CHECK WHETHER ENOUGH DATA EXISTS
    # ========================================================

    def can_sample(self, batch_size):

        return len(self.buffer) >= batch_size


# ============================================================
# TEST REPLAY BUFFER
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("DQN REPLAY BUFFER TEST")
    print("=" * 70)


    # --------------------------------------------------------
    # Create buffer
    # --------------------------------------------------------

    buffer = ReplayBuffer(
        capacity=100
    )

    print(
        f"\nBuffer capacity: "
        f"{buffer.capacity}"
    )


    # --------------------------------------------------------
    # Add sample experiences
    # --------------------------------------------------------

    for i in range(10):

        state = [
            0.5 + i * 0.01,
            0.6,
            0.1,
            0.2
        ]

        action = i % 3

        reward = (
            0.01 * i
        )

        next_state = [
            0.5,
            0.6 + i * 0.01,
            0.2,
            0.3
        ]

        done = (
            i == 9
        )

        buffer.push(
            state,
            action,
            reward,
            next_state,
            done
        )


    print(
        f"Experiences stored: "
        f"{len(buffer)}"
    )


    # --------------------------------------------------------
    # Sample experiences
    # --------------------------------------------------------

    batch_size = 4

    batch = buffer.sample(
        batch_size
    )

    print(
        f"\nSampled batch size: "
        f"{len(batch)}"
    )


    # --------------------------------------------------------
    # Display one experience
    # --------------------------------------------------------

    experience = batch[0]

    (
        state,
        action,
        reward,
        next_state,
        done
    ) = experience

    print("\nExample experience:")

    print(
        f"State      : {state}"
    )

    print(
        f"Action     : {action}"
    )

    print(
        f"Reward     : {reward:.4f}"
    )

    print(
        f"Next state : {next_state}"
    )

    print(
        f"Done       : {done}"
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    print("\n" + "-" * 70)
    print("REPLAY BUFFER VALIDATION")
    print("-" * 70)


    # Validate storage

    if len(buffer) == 10:

        print(
            "PASS: Experiences stored correctly."
        )

    else:

        print(
            "FAIL: Incorrect number of experiences."
        )


    # Validate sampling

    if len(batch) == batch_size:

        print(
            "PASS: Random batch sampling works."
        )

    else:

        print(
            "FAIL: Incorrect batch size."
        )


    # Validate experience structure

    if len(experience) == 5:

        print(
            "PASS: Experience contains "
            "(state, action, reward, next_state, done)."
        )

    else:

        print(
            "FAIL: Incorrect experience structure."
        )


    # Validate can_sample

    if buffer.can_sample(4):

        print(
            "PASS: can_sample() works correctly."
        )

    else:

        print(
            "FAIL: can_sample() returned incorrect result."
        )


    print("\n" + "=" * 70)
    print("REPLAY BUFFER TEST COMPLETED")
    print("=" * 70)