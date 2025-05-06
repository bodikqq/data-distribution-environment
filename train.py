import warnings
warnings.filterwarnings('ignore')
import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from custom_env_v3 import GraphEnv
import usefull_arrays as usfl_arr
import numpy as np

def test_env_manually(env):
    """Test the environment manually without the env checker."""
    print("\nTesting environment manually...")
    
    # Test reset
    print("Testing reset...")
    obs, info = env.reset()
    print("Reset successful. Observation keys:", obs.keys())
    print("Graph vertices count:", len(obs["graph"]["vertices"]))
    
    # Test step with empty action
    print("\nTesting step with empty action...")
    obs, reward, done, info = env.step([])
    print("Step successful.")
    print(f"Reward: {reward}")
    print(f"Info: {info}")
    
    # Test light control scenario with multiple random steps
    print("\nTesting light control scenario with multiple random steps...")

    
    total_steps = 5  # Reduced total steps but with more actions per step
    actions_per_step = 10  # Use 100 actions per step
    print(f"\nRunning {total_steps} steps with {actions_per_step} random actions each for lights scenario:")
    rewards = []
    
    for i in range(total_steps): 
        env.lights_scenario("4")
        # Generate multiple random actions for this step
        actions = []
        for _ in range(actions_per_step):
            actions.append([str(np.random.choice(usfl_arr.sensors)), 
                           str(np.random.choice(usfl_arr.controllers))])
        
        print(f"Step {i+1}, Number of actions: {len(actions)}")
        # For debugging, print a few sample actions
        if len(actions) > 3:
            print(f"Sample actions: {actions[:3]} ... (and {len(actions)-3} more)")
        else:
            print(f"Actions: {actions}")
        
        obs, reward, done, info = env.step(actions)
        rewards.append(reward)
        
        print(f"Reward: {reward}")
        print(f"Info: {info}")
        print(f"Environment reset: {obs.keys() is not None}")
        print("-" * 50)
        env.reset()
    
    print("\nRewards summary:")
    print(f"Rewards: {rewards}")
    print(f"Min reward: {min(rewards)}, Max reward: {max(rewards)}")
    print(f"Are all rewards the same? {len(set(rewards)) == 1}")
    
    return True

if __name__ == "__main__":
    try:
        # Create and initialize the environment
        env = GraphEnv(usfl_arr.descriptions_for_regular_tasks)
        
        print("Created environment successfully")
        print("Action space:", env.action_space)
        print("Observation space:", env.observation_space)
        
        # Try manual testing first
        if test_env_manually(env):
            print("\nManual testing successful!")
        
    except Exception as e:
        print("\nError in testing:", str(e))
        import traceback
        traceback.print_exc()