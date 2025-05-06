import warnings
warnings.filterwarnings('ignore')
import gymnasium as gym
from custom_env_v3 import GraphEnv
import usefull_arrays as usfl_arr
import numpy as np
import traceback

def test_complex_scenario(env, num_runs=3, actions_per_run=5):
    """Test the complex light scenario multiple times with additional random actions."""
    print(f"\nTesting complex light scenario for {num_runs} runs...")

    all_run_rewards = []

    for run in range(num_runs):
        print(f"\n===== Run {run + 1}/{num_runs} =====")

        # Test reset before each run
        print("Resetting environment...")
        obs, info = env.reset()
        print("Reset successful.")

        # Define the scenario parameters (can be varied per run if needed)
        room_ids_to_test = ["room8", "room9", "room10"] # Example rooms

        print(f"\nInitiating complex light scenario for rooms: {room_ids_to_test} from controller 4")

        random_actions = []
        if actions_per_run > 0:
            print(f"\nGenerating {actions_per_run} random regular task actions for this run...")
            for _ in range(actions_per_run):
                random_actions.append([str(np.random.choice(usfl_arr.sensors)), 
                           str(np.random.choice(usfl_arr.controllers))])
        

        # Run the environment step to process the scenario and random actions
        all_times = []
        print(f"\nProcessing scenario tasks and {len(random_actions)} additional actions using env.step()...")
        try:
            # Pass the generated random actions to the step function
            obs, reward, done, info = env.step(random_actions)
            print("Step successful after scenario initiation.")
            print(f"-> Reward for this run: {reward}")
            print(f"-> Info for this run: {info}")
            all_run_rewards.append(reward)
            all_times.append(info["time"])

            # --- Post-Scenario Check ---
            print("\n--- Post-Scenario Light State ---")
            for room_name in room_ids_to_test:
                if room_name in usfl_arr.rooms:
                    light_ids = usfl_arr.rooms[room_name].get("lights", [])
                    print(f"Room: {room_name}")
                    for light_id in light_ids:
                        light_vertex = env.find_edge_vertex(str(light_id))
                        if light_vertex:
                            print(f"  Light {light_id}: isOn={light_vertex.get('isOn', 'N/A')}, brightness={light_vertex.get('brightness', 'N/A')}")
                        else:
                            print(f"  Light {light_id}: Not found in graph")
                else:
                    print(f"Room: {room_name} - Not defined in usefull_arrays")
            print("----------------------------------")

        except Exception as e:
            print(f"Error during env.step() in run {run + 1}: {e}")
            traceback.print_exc()
            continue # Skip to the next run

    print("\n===== Testing Summary =====")
    if all_run_rewards:
        print(f"Rewards across {len(all_run_rewards)} runs: {all_run_rewards}")
        print(f"Average reward: {np.mean(all_run_rewards):.4f}")
        print(f"Min reward: {min(all_run_rewards):.4f}")
        print(f"Max reward: {max(all_run_rewards):.4f}")
        print(f"Times: {all_times}")
        print(f"Are all rewards the same? {len(set(all_run_rewards)) == 1}")
    else:
        print("No runs completed successfully.")

    return len(all_run_rewards) > 0 # Return True if at least one run succeeded

if __name__ == "__main__":
    try:
        # Create and initialize the environment
        env = GraphEnv(usfl_arr.descriptions_for_regular_tasks, scenario="complex_light")
        print("Created environment successfully for complex light test")

        # Run the complex scenario test multiple times
        num_test_runs = 3 # How many times to run the scenario
        num_random_actions = 200 # How many extra random tasks to add each run
        if test_complex_scenario(env, num_runs=num_test_runs, actions_per_run=num_random_actions):
            print("\nComplex scenario testing completed.")
        else:
            print("\nComplex scenario testing failed (no runs completed successfully).")

    except Exception as e:
        print("\nError during testing setup or execution:", str(e))
        traceback.print_exc()

