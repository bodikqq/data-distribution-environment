import warnings
warnings.filterwarnings('ignore')
import gymnasium as gym
from custom_env_v3 import GraphEnv
import usefull_arrays as usfl_arr
import numpy as np
import traceback
import time # Import the time module
import os # Import the os module

def test_complex_scenario(env, num_runs=3, actions_per_run=5):
    """Test the complex light scenario multiple times with additional random actions."""
    start_time = time.time() # Record start time
    print(f"\nTesting complex light scenario for {num_runs} runs...")

    all_run_rewards = []
    all_times = []
    all_run_results = [] # To store (reward, time) pairs
    for run in range(num_runs):
        print(f"\n===== Run {run + 1}/{num_runs} =====")

        # Test reset before each run
        print("Resetting environment...")
        obs, info = env.reset()
        print("Reset successful.")

        # Define the scenario parameters (can be varied per run if needed)
        room_ids_to_test = ["room8", "room10","room11","room3","room4","room5","room6","room7","room13","room14","room15","room16","room17","room18"]

        print(f"\nInitiating complex light scenario for rooms: {room_ids_to_test} from controller 4")

        random_actions = []
        if actions_per_run > 0:
            print(f"\nGenerating {actions_per_run} random regular task actions for this run...")
            for _ in range(actions_per_run):
                random_actions.append([str(np.random.choice(usfl_arr.sensors)), 
                           str(np.random.choice(usfl_arr.controllers))])
        

        # Run the environment step to process the scenario and random actions
        
        print(f"\nProcessing scenario tasks and {len(random_actions)} additional actions using env.step()...")
        try:
            # Pass the generated random actions to the step function
            obs, reward, done, info = env.step(random_actions)
            print("Step successful after scenario initiation.")
            print(f"-> Reward for this run: {reward}")
            print(f"-> Info for this run: {info}")
            all_run_rewards.append(reward)
            all_times.append(info["time"])
            all_run_results.append((reward, info["time"])) # Store as a pair

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
        print(f"Average reward: {np.mean(all_run_rewards):.3f}")
        print(f"Min reward: {min(all_run_rewards):.3f}")
        print(f"Max reward: {max(all_run_rewards):.3f}")
        print(f"Reward and Time pairs for each run:")
        for i, (r, t) in enumerate(all_run_results):
            print(f"  Run {i+1}: Reward={r:.3f}, Time={t:.3f}s")

        if len(all_run_rewards) >= 10: # Keep condition based on all_run_rewards as per previous logic
            try:
                output_dir = "test2_Results"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = os.path.join(output_dir, f"all_run_results_{timestamp}.txt") 
                with open(filename, "w") as f:
                    f.write("Reward,Time\n") 
                    for reward_value, time_value in all_run_results: 
                        f.write(f"{reward_value:.3f},{time_value}\n") 
                print(f"Successfully wrote {len(all_run_results)} run results to {filename}")
            except Exception as e:
                print(f"Error writing run results to file: {e}")
    else:
        print("No runs completed successfully.")

    end_time = time.time() # Record end time
    execution_time = end_time - start_time
    print(f"\nTotal execution time for test_complex_scenario: {execution_time} seconds")

    return len(all_run_rewards) > 0 # Return True if at least one run succeeded

if __name__ == "__main__":
    try:
        # Create and initialize the environment
        env = GraphEnv(usfl_arr.descriptions_for_regular_tasks, scenario="complex_light")
        print("Created environment successfully for complex light test")

        # Run the complex scenario test multiple times
        num_test_runs = 1 # How many times to run the scenario
        num_random_actions = 200 # How many extra random tasks to add each run
        if test_complex_scenario(env, num_runs=num_test_runs, actions_per_run=num_random_actions):
            print("\nComplex scenario testing completed.")
        else:
            print("\nComplex scenario testing failed (no runs completed successfully).")

    except Exception as e:
        print("\nError during testing setup or execution:", str(e))
        traceback.print_exc()

