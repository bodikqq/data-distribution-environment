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
        observation, reward, done, info = environment.step(action_matrix)
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
        
        # Print status of 3 lights
        print("\n--- Devices Status ---")
        print("Light Status:")
        # Select 3 light IDs from different rooms
        light_ids = [usfl_arr.rooms['room3']['lights'][0], usfl_arr.rooms['room8']['lights'][0], usfl_arr.rooms['room13']['lights'][0]]
        for light_id in light_ids:
            light_vertex = find_vertex_by_id(observation['graph']['vertices'], light_id)
            if light_vertex:
                is_on = "ON" if light_vertex.get('isOn', 0) == 1 else "OFF"
                brightness = light_vertex.get('brightness', 'N/A')
                print(f"  Light {light_id}: {is_on}, Brightness: {brightness}")
            else:
                print(f"  Light {light_id}: Not found")
        
        # Print status of 3 ventilations
        print("\nVentilation Status:")
        ventilation_ids = [usfl_arr.rooms['room3']['ventelation'], usfl_arr.rooms['room8']['ventelation'], usfl_arr.rooms['room13']['ventelation']]
        for ventilation_id in ventilation_ids:
            ventilation_vertex = find_vertex_by_id(observation['graph']['vertices'], ventilation_id)
            if ventilation_vertex:
                is_on = "ON" if ventilation_vertex.get('isOn', 0) == 1 else "OFF"
                fan_speed = ventilation_vertex.get('fan_speed', 'N/A')
                print(f"  Ventilation {ventilation_id}: {is_on}, Fan Speed: {fan_speed}")
            else:
                print(f"  Ventilation {ventilation_id}: Not found")
        
        # Print status of 3 doors
        print("\nDoor Status:")
        door_ids = [usfl_arr.rooms['room3']['door'], usfl_arr.rooms['room8']['door'], usfl_arr.rooms['room13']['door']]
        for door_id in door_ids:
            door_vertex = find_vertex_by_id(observation['graph']['vertices'], door_id)
            if door_vertex:
                locked = "LOCKED" if door_vertex.get('locked', 0) == 1 else "UNLOCKED"
                secure_mode = "ON" if door_vertex.get('secure_mode', 0) == 1 else "OFF"
                print(f"  Door {door_id}: {locked}, Secure Mode: {secure_mode}")
            else:
                print(f"  Door {door_id}: Not found")
    
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
    filename = os.path.join(output_dir, f"multi_sensor_scenario_{timestamp}.txt")
    
    # Write results to file
    with open(filename, "w") as f:
        f.write("="*70 + "\n")
        f.write("MULTI-SENSOR SECURITY SCENARIO RESULTS\n")
        f.write("="*70 + "\n")
        f.write(f"Test Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of runs: {num_steps}\n")
        f.write(f"Actions per run: {actions_per_step}\n")
        f.write(f"Total reward: {total_reward}\n")
        f.write(f"Average reward per run: {total_reward/num_steps if num_steps > 0 else 0}\n")
        f.write(f"Total execution time: {total_execution_time:.2f} seconds\n\n")
        
        f.write("Run Details:\n")
        f.write("Run,Reward,Time,Tasks,Steps,ExecutionTime\n")
        for run_info in all_steps_info:
            f.write(f"{run_info['run']},{run_info['reward']},{run_info['time']},"
                   f"{run_info['num_tasks']},{run_info['steps_taken']},{run_info['execution_time']:.2f}\n")
    
    print(f"\nResults saved to file: {filename}")
    
    # Create a simple plot of rewards
    create_reward_plot(all_steps_info, output_dir, timestamp)

def create_reward_plot(all_steps_info, output_dir, timestamp):
    """Create and save a plot of time vs rewards"""
    try:
        plt.figure(figsize=(10, 6))
        
        runs = [info["run"] for info in all_steps_info]
        rewards = [info["reward"] for info in all_steps_info]
        times = [info["time"] for info in all_steps_info]
        
        # Single plot: Environment Time (X) vs Reward (Y)
        plt.scatter(times, rewards, marker='o', color='blue', s=50)
        
        # Add run numbers as annotations
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
        if len(rewards) > 1:
            try:
                z = np.polyfit(times, rewards, 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(times), max(times), 100)
                plt.plot(x_range, p(x_range), "r--", alpha=0.8, label=f"y={z[0]:.3f}x+{z[1]:.3f}")
                plt.legend()
            except Exception as e:
                print(f"Could not draw trend line: {e}")
        
        plt.tight_layout()
        
        plot_filename = os.path.join(output_dir, f"multi_sensor_plot_{timestamp}.png")
        plt.savefig(plot_filename)
        plt.close()
        
        print(f"Time vs Reward plot saved to: {plot_filename}")
    except Exception as e:
        print(f"Error creating plot: {e}")

if __name__ == "__main__":
    # Run the test with 3 steps, 20 actions per step, and random actions enabled
    test_multi_sensor_scenario(num_steps=2, actions_per_step=700, use_random_actions=True, save_results=True)
    
    # Uncomment the line below to run without random actions
    # test_multi_sensor_scenario(num_steps=3, use_random_actions=False, save_results=True)