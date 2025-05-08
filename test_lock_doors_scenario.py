import custom_env_v3 as env
import usefull_arrays as usfl_arr
import numpy as np

def test_lock_doors_scenario():
    """
    Test the lock doors scenario where tasks are sent to lock all doors in the system.
    """
    print("Starting Lock Doors Scenario Test")
    
    # Create environment with standard settings
    environment = env.GraphEnv(usfl_arr.descriptions_for_regular_tasks)
    
    # Initialize the environment
    observation, info = environment.reset()
    
    # Get a controller ID to start from (using controller 4 as default)
    starting_controller_id = "4"
    
    # Run the initial time steps to gather sensor information
    for _ in range(20):
        environment.time_step()
    
    print(f"\nTriggering lock_doors_scenario from controller {starting_controller_id}")
    
    # Call the lock_doors_scenario function
    tasks, success = environment.lock_doors_scenario(starting_controller_id)
    
    if success:
        print(f"Successfully created {len(tasks)} door locking tasks")
    else:
        print("Failed to create door locking tasks")
        return
    
    # Process the tasks in the environment
    action_matrix = np.array([])
    observation, reward, done, info = environment.step(action_matrix)
    
    print("\n--- Test Results ---")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Number of tasks processed: {info.get('num_tasks', 'N/A')}")
    print(f"Time steps processed: {info.get('steps_taken', 'N/A')}")
    
    # Print summary
    print("\nLock Doors Scenario Test Complete!")
    print("The test simulates:")
    print(" 1. Sending tasks to lock all doors in the system")
    print(" 2. Each task sets the door's 'locked' parameter to 1 and 'secure_mode' to 1")
    print(f" 3. All tasks are dispatched from controller {starting_controller_id}")
    
    return observation, reward, done, info

if __name__ == "__main__":
    test_lock_doors_scenario()