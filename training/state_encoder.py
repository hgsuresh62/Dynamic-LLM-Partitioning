
import numpy as np


class StateEncoder:
    """
    Encodes telemetry and the current partition into
    a normalized five-feature RL state.
    """

    def __init__(self):
        self.total_layers = 28
        self.state_size = 5

    def encode(self, telemetry, split_point):
        worker_1 = float(
            telemetry["worker_1_vram_utilization"]
        ) / 100.0

        worker_2 = float(
            telemetry["worker_2_vram_utilization"]
        ) / 100.0

        network_latency = float(
            telemetry["network_latency_ms"]
        ) / 100.0

        token_position = float(
            telemetry["token_position"]
        )

        normalized_split = float(split_point) / self.total_layers

        state = np.array(
            [
                np.clip(worker_1, 0.0, 1.0),
                np.clip(worker_2, 0.0, 1.0),
                np.clip(network_latency, 0.0, 1.0),
                np.clip(token_position, 0.0, 1.0),
                np.clip(normalized_split, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

        return state

    def display(self, state):
        print("\nEncoded RL State:")
        print(f"Worker 1 utilization : {state[0]:.4f}")
        print(f"Worker 2 utilization : {state[1]:.4f}")
        print(f"Network latency       : {state[2]:.4f}")
        print(f"Token position        : {state[3]:.4f}")
        print(f"Current split         : {state[4]:.4f}")
        print(f"State shape           : {state.shape}")
        print(f"State dtype           : {state.dtype}")


if __name__ == "__main__":
    print("=" * 65)
    print("RL STATE ENCODER TEST")
    print("=" * 65)

    telemetry = {
        "worker_1_vram_utilization": 70.0,
        "worker_2_vram_utilization": 50.0,
        "network_latency_ms": 40.0,
        "token_position": 0.50,
    }

    encoder = StateEncoder()
    state = encoder.encode(
        telemetry=telemetry,
        split_point=12,
    )

    encoder.display(state)

    assert state.shape == (5,)
    assert state.dtype == np.float32
    assert np.all(state >= 0.0)
    assert np.all(state <= 1.0)
    assert np.isclose(state[4], 12 / 28)

    print("\nPASS: State contains five normalized features.")
    print("PASS: State uses float32.")
    print("PASS: Split point is encoded correctly.")
    print("=" * 65)
