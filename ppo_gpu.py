import os
import sys
# Disable tkinter usage completely and set backend to 'Agg' before any other imports
os.environ['MPLBACKEND'] = 'Agg'  # Force matplotlib to use Agg backend
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

import matplotlib
matplotlib.use('Agg')  # Redundant but ensure it's set

import time
import datetime
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from collections import deque

# Import environment and utilities
import usefull_arrays as usfl_arr
from custom_env_v3 import GraphEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

# Use GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")
torch_threads = 6 if device.type == 'cpu' else 1  # More CPU threads if no GPU available
torch.set_num_threads(torch_threads)
print(f"Using {torch_threads} CPU threads for tensor operations")

# Create directories for saving models, plots and logs
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
results_dir = f"ppo_gpu_results_{timestamp}"
models_dir = os.path.join(results_dir, "models")
logs_dir = os.path.join(results_dir, "logs")
plots_dir = results_dir

# Ensure directories exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Custom Feature Extractor for our specific action space
class CustomNetworkArchitecture(BaseFeaturesExtractor):
    """
    Custom feature extractor optimized for the large action space 
    in the Graph Environment.
    """
    def __init__(self, observation_space, features_dim=256):
        super(CustomNetworkArchitecture, self).__init__(observation_space, features_dim)
        
        # Simple but effective network for very simple observations
        self.features_extractor = torch.nn.Sequential(
            torch.nn.Linear(observation_space.shape[0], 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, features_dim),
            torch.nn.ReLU(),
        )

    def forward(self, observations):
        return self.features_extractor(observations)

# Custom callback with extended monitoring capabilities
class EnhancedTrainingCallback(BaseCallback):
    """
    Custom callback for monitoring training progress with extended features:
    - Tracks reward statistics
    - Creates progress plots
    - Saves models at regular intervals
    - Captures reward variance
    - Monitors action entropy
    """
    def __init__(self, 
                 check_freq=1000,
                 save_freq=5000, 
                 save_path=models_dir, 
                 timestamp=timestamp, 
                 verbose=1,
                 eval_env=None):
        super(EnhancedTrainingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_freq = save_freq
        self.save_path = save_path
        self.timestamp = timestamp
        self.eval_env = eval_env
        
        # Metrics tracking
        self.rewards = []
        self.steps = []
        self.episode_rewards = []
        self.reward_variance = []
        self.entropy_values = []
        self.last_mean_reward = 0
        self.training_start_time = None
        
        # More granular reward tracking
        self.reward_buffer = deque(maxlen=100)
        
        # Create a CSV file for saving reward data
        self.csv_path = os.path.join(plots_dir, f"reward_data_{self.timestamp}.csv")
        with open(self.csv_path, 'w') as f:
            f.write("step,mean_reward,trend_slope,trend_intercept\n")
        
    def _on_training_start(self):
        """
        Called at the start of training.
        """
        self.training_start_time = time.time()
        print("Starting PPO training...")
        
    def _on_step(self) -> bool:
        """
        Called at each step of training.
        """
        # Track steps
        self.steps.append(self.num_timesteps)
        
        # Access episode returns from VecMonitor
        if len(self.model.ep_info_buffer) > 0:
            # Get all completed episodes since last check
            new_rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer 
                           if ep_info["r"] is not None]
            
            if new_rewards:
                # Track the new rewards
                self.episode_rewards.extend(new_rewards)
                self.reward_buffer.extend(new_rewards)
                
                # Update reward statistics
                self.last_mean_reward = np.mean(new_rewards)
                if len(new_rewards) > 1:
                    self.reward_variance.append(np.var(new_rewards))
        
        # Always add the latest reward information
        self.rewards.append(self.last_mean_reward)
        
        # Track policy entropy if available
        if hasattr(self.model, "logger") and "train/entropy" in self.model.logger.name_to_value:
            self.entropy_values.append(self.model.logger.name_to_value["train/entropy"])
        
        # Create plot and log at check_freq intervals
        if self.num_timesteps % self.check_freq == 0:
            self._create_training_plot()
            
            # Log detailed statistics
            if len(self.reward_buffer) > 0:
                recent_rewards = np.array(self.reward_buffer)
                avg_reward = np.mean(recent_rewards)
                median_reward = np.median(recent_rewards)
                min_reward = np.min(recent_rewards)
                max_reward = np.max(recent_rewards)
                reward_std = np.std(recent_rewards)
                
                elapsed_time = time.time() - self.training_start_time
                steps_per_second = self.num_timesteps / elapsed_time
                
                print(f"Step: {self.num_timesteps} ({steps_per_second:.1f} steps/sec)")
                print(f"Recent rewards - Avg: {avg_reward:.2f}, Med: {median_reward:.2f}, Range: [{min_reward:.2f}, {max_reward:.2f}], Std: {reward_std:.2f}")
            
        # Save model at save_freq intervals
        if self.num_timesteps % self.save_freq == 0:
            model_path = os.path.join(
                self.save_path, 
                f"ppo_gpu_model_{self.timestamp}_{self.num_timesteps}_steps.zip"
            )
            self.model.save(model_path)
            print(f"Model checkpoint saved: {model_path}")
            
        return True
        
    def _create_training_plot(self):
        """Create comprehensive training progress plots and save data."""
        if len(self.rewards) > 0:
            plt.figure(figsize=(12, 6))
            
            # Skip the first 10 values for better visualization
            skip_values = min(10, len(self.rewards) // 10)  # Dynamic skip based on data size
            
            if len(self.rewards) <= skip_values:
                iterations = list(range(len(self.rewards)))
                plot_rewards = self.rewards
            else:
                iterations = list(range(skip_values, len(self.rewards)))
                plot_rewards = self.rewards[skip_values:]
            
            # Main reward plot with regression line
            plt.plot(iterations, plot_rewards, 'b-', linewidth=1.5, label='Mean Reward')
            
            # Add regression line if we have enough data points
            trend_slope = 0
            trend_intercept = 0
            if len(iterations) > 2:
                # Calculate regression line
                z = np.polyfit(iterations, plot_rewards, 1)
                trend_slope = z[0]
                trend_intercept = z[1]
                p = np.poly1d(z)
                x_range = np.linspace(min(iterations), max(iterations), 100)
                plt.plot(x_range, p(x_range), "r--", linewidth=1.5, 
                         label=f'Trend: y={trend_slope:.4f}x+{trend_intercept:.2f}')
                plt.legend()
            
            plt.title(f'Reward Progress (Step {self.num_timesteps})')
            plt.xlabel('Training Iterations')
            plt.ylabel('Mean Episode Reward')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            # Save plot - just update a single file instead of creating many
            plt.savefig(os.path.join(plots_dir, "training_progress.png"))
            # Also save as latest_progress.png for compatibility
            plt.savefig(os.path.join(plots_dir, "latest_progress.png"))
            plt.close()
            
            # Save reward data to CSV
            with open(self.csv_path, 'a') as f:
                f.write(f"{self.num_timesteps},{self.last_mean_reward},{trend_slope},{trend_intercept}\n")
            
            # Save numerical data for further analysis (just update the same file)
            np.savez(
                os.path.join(plots_dir, f"training_data_{self.timestamp}.npz"),
                steps=np.array(self.steps),
                rewards=np.array(self.rewards),
                episode_rewards=np.array(self.episode_rewards),
                reward_variance=np.array(self.reward_variance),
                entropy_values=np.array(self.entropy_values)
            )

def make_env(rank, seed=0):
    """
    Create an environment with a specific rank seed.
    Useful for parallel environments.
    """
    def _init():
        # Create environment with the same scenario as your original implementation
        env = GraphEnv(
            usfl_arr.descriptions_for_regular_tasks, 
            json_path="graph_output.json", 
            scenario="multi_sensor_scenario"
        )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    # Environment setup
    num_cpu = 8 # Use multiple CPUs if available, leave one for system
    print(f"Creating {num_cpu} parallel environments")
    
    if num_cpu > 1:
        # Parallel environments for faster training
        env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    else:
        # Single environment
        env = DummyVecEnv([make_env(0)])
    
    # Wrap with monitor to track episode returns
    env = VecMonitor(env)
    
    # Enhanced PPO model configuration
    model = PPO(
    policy="MlpPolicy",
    env=env,
    device=device
)


    # Create callback
    callback = EnhancedTrainingCallback()
    
    # Training parameters - much longer than original for proper learning
    total_timesteps = 200000 
    
    print(f"Starting training for {total_timesteps} steps...")
    start_time = time.time()
    # Train the model
    model.learn(
        total_timesteps=total_timesteps, 
        callback=callback,
        progress_bar=True
    )
    
    # Report training time
    training_duration = time.time() - start_time
    hours, remainder = divmod(training_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training completed in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    
    # Save final model
    final_model_path = os.path.join(models_dir, f"ppo_gpu_final_{timestamp}.zip")
    model.save(final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    # Quick test of the model
    print("Testing the model with a sample observation...")
    obs = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    print(f"Predicted action shape: {action.shape}")
    
    # Close environment
    env.close()