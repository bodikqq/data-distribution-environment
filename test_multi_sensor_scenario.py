import custom_env_v3 as env
import usefull_arrays as usfl_arr
import random
import numpy as np
import time
import os
import datetime
import matplotlib.pyplot as plt

def test_multi_sensor_scenario(num_steps=1, actions_per_step=5, use_random_actions=False, save_results=True):
    """
    Test the multi-sensor scenario where a light task is sent with confirmations from
    CO2, movement, LiDAR, and noise sensors. When confirmed, it will trigger all lights to turn on.
    
    Parameters:
    - num_steps (int): Number of steps to run in the environment
    - actions_per_step (int): Number of random actions to generate per step
    - use_random_actions (bool): Whether to use random actions during the steps
    - save_results (bool): Whether to save results to a file
    """
    start_time = time.time()  # Record overall start time
    print("Starting Multi-Sensor Scenario Test")
    
    total_reward = 0
    all_steps_info = []
    
    # Run multiple steps by resetting the environment each time
    for step in range(num_steps):
        step_start_time = time.time()  # Record step start time
        print(f"\n{'='*50}")
        print(f"EXECUTING RUN {step+1}/{num_steps}")
        print(f"{'='*50}\n")
        
        # Create a fresh environment for each step
        environment = env.GraphEnv(usfl_arr.descriptions_for_regular_tasks, scenario="multi_sensor_scenario")
        
        # Initialize the environment
        observation, info = environment.reset()
        
        if use_random_actions:
            # Generate random actions similar to test2.py
            random_actions = []
            print(f"\nGenerating {actions_per_step} random regular task actions for this run...")
            for _ in range(actions_per_step):
                random_actions.append([
                    str(np.random.choice(usfl_arr.sensors)), 
                    str(np.random.choice(usfl_arr.controllers))
                ])
            print(f"Random actions generated: {len(random_actions)} actions")
            action_matrix = random_actions
        else:
            # For this test without random actions, we don't need to add any specific tasks
            action_matrix = []
            print("Using empty action list for this run")
        
        # Run the environment step
        observation, reward, done, truncated, info = environment.step(action_matrix)
        step_time = time.time() - step_start_time  # Calculate step execution time
        total_reward += reward
        
        print(f"Run reward: {reward}")
        print(f"Run info: {info}")
        print(f"Run execution time: {step_time:.2f} seconds")
        
        all_steps_info.append({
            "run": step+1,
            "reward": reward,
            "time": info.get("time", 0),
            "num_tasks": info.get("num_tasks", 0),
            "steps_taken": info.get("steps_taken", 0),
            "execution_time": step_time
        })
        
        # Print status of devices for display only
        print("\n--- Devices Status ---")
        print("Note: Direct access to device status is not available with the constant observation format.")
        print("Light Status: Not available in current observation format")
        print("Ventilation Status: Not available in current observation format")
        print("Door Status: Not available in current observation format")
    
    total_execution_time = time.time() - start_time  # Calculate total execution time
    
    # Print summary of all runs
    print("\n" + "="*70)
    print("MULTI-SENSOR SCENARIO TEST SUMMARY")
    print("="*70)
    print(f"Total runs: {num_steps}")
    print(f"Total reward across all runs: {total_reward}")
    print(f"Average reward per run: {total_reward/num_steps if num_steps > 0 else 0}")
    print(f"Total execution time: {total_execution_time:.2f} seconds")
    
    print("\nRun Details:")
    for run_info in all_steps_info:
        print(f"  Run {run_info['run']}: Reward={run_info['reward']}, " 
              f"Time={run_info['time']}, Tasks={run_info['num_tasks']}, "
              f"Steps={run_info['steps_taken']}, "
              f"Execution time={run_info['execution_time']:.2f}s")
    
    print("\nThe multi-sensor scenario tested:")
    print(" 1. Sending a light task with confirmations required from multiple sensor types")
    print(" 2. The confirmations have conditions which are always true")
    print(" 3. When the task is confirmed, it triggers the lights_scenario to turn on all lights")
    
    # Save results to file
    if save_results:
        save_results_to_file(all_steps_info, num_steps, actions_per_step, total_reward, total_execution_time)
    
    return all_steps_info

def find_vertex_by_id(vertices, vertex_id):
    """Helper function to find a vertex by its ID in the graph"""
    for vertex in vertices:
        if str(vertex['id']) == str(vertex_id):
            return vertex
    return None

def save_results_to_file(all_steps_info, num_steps, actions_per_step, total_reward, total_execution_time):
    """Save test results to a file"""
    # Create directory if it doesn't exist
    output_dir = "security_scenario_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Generate timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #filename = os.path.join(output_dir, f"multi_sensor_scenario_{timestamp}.txt")
    
    # Write results to file
    #with open(filename, "w") as f:
    #    f.write("="*70 + "\n")
    #    f.write("MULTI-SENSOR SECURITY SCENARIO RESULTS\n")
    #    f.write("="*70 + "\n")
    #    f.write(f"Test Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    #    f.write(f"Number of runs: {num_steps}\n")
    #    f.write(f"Actions per run: {actions_per_step}\n")
    #    f.write(f"Total reward: {total_reward}\n")
    #    f.write(f"Average reward per run: {total_reward/num_steps if num_steps > 0 else 0}\n")
    #    f.write(f"Total execution time: {total_execution_time:.2f seconds\n\n")
    #    
    #    f.write("Run Details:\n")
    #    f.write("Run,Reward,Time,Tasks,Steps,ExecutionTime\n")
    #    for run_info in all_steps_info:
    #        f.write(f"{run_info['run']},{run_info['reward']},{run_info['time']},"
    #               f"{run_info['num_tasks']},{run_info['steps_taken']},{run_info['execution_time']:.2f}\n")
    #
    #print(f"\nResults saved to file: {filename}")
    
    # Create a simple plot of rewards
    create_reward_plot(all_steps_info, output_dir, timestamp,actions_per_step)

def create_reward_plot(all_steps_info, output_dir, timestamp, actions_per_step):
    """Create and save plots of rewards vs episodes and time vs rewards"""
    try:
        # Create two plots: Episodes vs Rewards and Time vs Rewards
        plt.figure(figsize=(10, 6))
        
        runs = [info["run"] for info in all_steps_info]
        rewards = [info["reward"] for info in all_steps_info]
        times = [info["time"] for info in all_steps_info]          # Plot 1: Episodes (runs) vs Rewards - similar to ppo_gpu style
        # Skip the first few values for better visualization if there are many data points
        skip_values = min(10, len(rewards) // 10) if len(rewards) > 30 else 0  # Dynamic skip based on data size
        
        if len(rewards) <= skip_values:
            plot_runs = runs
            plot_rewards = rewards
        else:
            plot_runs = runs[skip_values:]
            plot_rewards = rewards[skip_values:]              # Main reward plot with clean lines (no markers) and transparency
        plt.plot(plot_runs, plot_rewards, 'b-', linewidth=1.5, alpha=0.47, label='Reward')
        
        # Add regression line
        if len(plot_rewards) >= 2:
            try:
                z = np.polyfit(plot_runs, plot_rewards, 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(plot_runs), max(plot_runs), 100)
                plt.plot(x_range, p(x_range), "r--", linewidth=1.5, 
                         label=f'Trend: y={z[0]:.4f}x+{z[1]:.2f}')
            except Exception as e:
                print(f"Could not draw regression line: {e}")
        
        # Add moving average
        window_size = 30  # Adjust window size based on data length
        if len(rewards) >= window_size:
            try:
                moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                # Align moving average with original data points
                start_idx = window_size // 2
                end_idx = start_idx + len(moving_avg)
                plt.plot(runs[start_idx:end_idx], moving_avg, color='darkblue', linewidth=2, label=f'Moving Avg (n={window_size})')
            except Exception as e:
                print(f"Could not calculate moving average: {e}")
                
        # Show legend with all plot elements
        plt.legend()
        
        plt.title('Episodes vs Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save episodes vs rewards plot
        episodes_plot_filename = os.path.join(output_dir, f"episodes_vs_rewards_{actions_per_step}_actions.png")
        plt.savefig(episodes_plot_filename)
        plt.close()
        
        print(f"Episodes vs Reward plot saved to: {episodes_plot_filename}")
        
        # Plot 2: Environment Time vs Rewards (original plot)
        plt.figure(figsize=(10, 6))
        plt.scatter(times, rewards, marker='o', color='green', s=50)
        
        # Add annotations if there aren't too many points
        if len(runs) <= 20:
            for i, run in enumerate(runs):
                plt.annotate(f'Run {run}', (times[i], rewards[i]), 
                            textcoords="offset points", 
                            xytext=(0,10), 
                            ha='center')
        
        plt.title('Multi-Sensor Security Scenario - Environment Time vs Reward')
        plt.xlabel('Environment Time')
        plt.ylabel('Reward')
        plt.grid(True)
        
        # Draw a best fit line if there are enough points
        if len(rewards) >= 10:
            try:
                z = np.polyfit(times, rewards, 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(times), max(times), 100)
                plt.plot(x_range, p(x_range), "r--", alpha=0.8, label=f"Trend: y={z[0]:.3f}x+{z[1]:.3f}")
                plt.legend()
            except Exception as e:
                print(f"Could not draw trend line: {e}")
        
        plt.tight_layout()
        
        # Save time vs rewards plot (original)
        time_plot_filename = os.path.join(output_dir, f"time_vs_rewards_{actions_per_step}_actions.png")
        plt.savefig(time_plot_filename)
        plt.close()
        
        print(f"Time vs Reward plot saved to: {time_plot_filename}")
    except Exception as e:
        print(f"Error creating plots: {e}")

if __name__ == "__main__":
    # Run the test with 3 steps, 20 actions per step, and random actions enabled
    test_multi_sensor_scenario(num_steps=10000, actions_per_step=300, use_random_actions=True, save_results=True)
    
    # Uncomment the line below to run without random actions
    # test_multi_sensor_scenario(num_steps=3, use_random_actions=False, save_results=True)