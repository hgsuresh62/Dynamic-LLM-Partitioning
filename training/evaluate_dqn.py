
import random
import torch

from training.dqn_agent import DQN
from training.rl_environment import PartitioningEnvironment


MODEL_PATH = "models/dqn_policy_5features.pt"
STATE_SIZE = 5
ACTION_SIZE = 3
STARTING_SPLIT = 12
NUM_TEST_STATES = 100
SEED = 42

ACTION_NAMES = {
    0: "Keep Split",
    1: "Shift Left",
    2: "Shift Right",
}


def generate_test_states():
    """Generate reproducible, varied simulated telemetry states."""
    rng = random.Random(SEED)
    states = []

    for _ in range(NUM_TEST_STATES):
        worker_1 = rng.uniform(0.40, 0.80)
        worker_2 = rng.uniform(0.40, 0.80)
        network_latency = rng.uniform(0.02, 1.00)
        token_position = rng.uniform(0.10, 1.00)
        normalized_split = STARTING_SPLIT / 28

        states.append([
            worker_1,
            worker_2,
            network_latency,
            token_position,
            normalized_split,
        ])

    return states


def load_model():
    """Load the saved five-feature DQN checkpoint."""
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    agent = DQN(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        saved_state_size = checkpoint.get("state_size")

        if saved_state_size != STATE_SIZE:
            raise ValueError(
                f"Expected {STATE_SIZE} features, "
                f"but checkpoint reports {saved_state_size}."
            )

        agent.load_state_dict(checkpoint["model_state_dict"])
    else:
        agent.load_state_dict(checkpoint)

    agent.eval()
    return agent


def main():
    print("=" * 72)
    print("DQN GENERALIZATION TEST - 100 SIMULATED STATES")
    print("=" * 72)

    env = PartitioningEnvironment()
    agent = load_model()
    states = generate_test_states()

    print(f"Checkpoint: {MODEL_PATH}")
    print(f"Test states: {len(states)}")
    print(f"Starting split: {STARTING_SPLIT}/"
          f"{env.total_layers - STARTING_SPLIT}")
    print("Exploration: Disabled (greedy Q-value selection)")
    print()

    matches = 0
    total_selected_reward = 0.0
    total_best_reward = 0.0
    total_regret = 0.0
    action_counts = {action: 0 for action in range(ACTION_SIZE)}

    for index, state in enumerate(states, start=1):
        state_tensor = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = agent(state_tensor).squeeze(0)

        dqn_action = int(torch.argmax(q_values).item())
        action_counts[dqn_action] += 1

        old_time = env._estimate_execution_time(
            STARTING_SPLIT, state
        )

        candidate_results = []

        for action in range(ACTION_SIZE):
            if action == 0:
                candidate_split = STARTING_SPLIT
            elif action == 1:
                candidate_split = STARTING_SPLIT - 1
            else:
                candidate_split = STARTING_SPLIT + 1

            candidate_time = env._estimate_execution_time(
                candidate_split, state
            )

            candidate_reward = env._calculate_reward(
                old_time,
                candidate_time,
                action,
            )

            candidate_results.append({
                "action": action,
                "split": candidate_split,
                "time": candidate_time,
                "reward": candidate_reward,
            })

        selected = candidate_results[dqn_action]
        best_reward = max(
            result["reward"] for result in candidate_results
        )

        # Allow ties when comparing with the simulator.
        best_actions = [
            result["action"]
            for result in candidate_results
            if abs(result["reward"] - best_reward) < 1e-9
        ]

        if dqn_action in best_actions:
            matches += 1

        regret = max(0.0, best_reward - selected["reward"])

        total_selected_reward += selected["reward"]
        total_best_reward += best_reward
        total_regret += regret

        if index <= 10:
            print(
                f"Test {index:03d} | "
                f"DQN: {ACTION_NAMES[dqn_action]:<12} | "
                f"Reward: {selected['reward']:+.4f} | "
                f"Best reward: {best_reward:+.4f} | "
                f"{'MATCH' if dqn_action in best_actions else 'MISMATCH'}"
            )

    match_rate = matches / NUM_TEST_STATES * 100
    average_selected_reward = total_selected_reward / NUM_TEST_STATES
    average_best_reward = total_best_reward / NUM_TEST_STATES
    average_regret = total_regret / NUM_TEST_STATES

    print("\n" + "=" * 72)
    print("100-STATE EVALUATION SUMMARY")
    print("=" * 72)
    print(f"States tested:                 {NUM_TEST_STATES}")
    print(f"Best-action matches:           {matches}/{NUM_TEST_STATES}")
    print(f"Best-action match rate:        {match_rate:.2f}%")
    print(f"Average DQN reward:            {average_selected_reward:+.5f}")
    print(f"Average best possible reward:  {average_best_reward:+.5f}")
    print(f"Average reward regret:         {average_regret:.5f}")

    print("\nDQN action distribution:")
    for action in range(ACTION_SIZE):
        count = action_counts[action]
        percentage = count / NUM_TEST_STATES * 100

        print(
            f"  {ACTION_NAMES[action]:<12}: "
            f"{count:3d} ({percentage:5.1f}%)"
        )

    print("\nInterpretation:")
    print("- Match rate measures agreement with the simulator's best")
    print("  immediate action for these generated states.")
    print("- Reward regret measures the gap from that best action.")
    print("- These are simulated comparisons, not real Llama benchmarks.")
    print("=" * 72)


if __name__ == "__main__":
    main()
