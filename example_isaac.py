#!/usr/bin/env python3

"""
Example script for using gym-dso100 with Isaac Gym backend.

This script demonstrates how to:
1. Create and use the Isaac Gym version of the lift environment
2. Run a simple random policy
3. Monitor performance and success rates
4. Handle multiple parallel environments
"""

import os
import sys
import numpy as np
import torch

# Add the gym_dso100 package to path
sys.path.append(os.path.join(os.path.dirname(__file__), "gym_dso100"))

from gym_dso100.tasks.lift_isaac import LiftIsaac


def main():
    """Main function to demonstrate Isaac Gym environment usage."""
    
    print("=" * 60)
    print("gym-dso100 Isaac Gym Example")
    print("=" * 60)
    
    # Environment configuration
    config = {
        "obs_type": "state",  # Options: "state", "pixels", "depth"
        "render_mode": "human",  # Options: "human", "rgb_array"
        "num_envs": 4,  # Number of parallel environments
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "headless": False,  # Set to True for headless operation
    }
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # Create environment
        print("Creating Isaac Gym environment...")
        env = LiftIsaac(**config)
        print(f"Environment created successfully!")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        print()
        
        # Reset environment
        print("Resetting environment...")
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        print()
        
        # Run random policy for demonstration
        print("Running random policy...")
        episode_rewards = np.zeros(config["num_envs"])
        episode_lengths = np.zeros(config["num_envs"])
        success_count = 0
        total_episodes = 0
        
        for step in range(1000):  # Run for 1000 steps
            # Random actions
            actions = env.action_space.sample()
            if config["num_envs"] > 1:
                actions = np.array([env.action_space.sample() for _ in range(config["num_envs"])])
            
            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Accumulate rewards
            episode_rewards += rewards
            episode_lengths += 1
            
            # Check for episode completion
            done = terminated | truncated
            if np.any(done):
                completed_envs = np.where(done)[0]
                
                for env_idx in completed_envs:
                    total_episodes += 1
                    
                    # Check if successful
                    if hasattr(env, 'is_success'):
                        success = env.is_success()[env_idx]
                        if success:
                            success_count += 1
                    
                    print(f"Episode {total_episodes} completed:")
                    print(f"  Environment: {env_idx}")
                    print(f"  Reward: {episode_rewards[env_idx]:.2f}")
                    print(f"  Length: {episode_lengths[env_idx]}")
                    if hasattr(env, 'is_success'):
                        print(f"  Success: {success}")
                    print()
                    
                    # Reset episode tracking for completed environments
                    episode_rewards[env_idx] = 0
                    episode_lengths[env_idx] = 0
            
            # Render environment
            if config["render_mode"] == "human":
                env.render()
            
            # Print progress every 100 steps
            if (step + 1) % 100 == 0:
                success_rate = success_count / max(total_episodes, 1) * 100
                print(f"Step {step + 1}/1000 | Episodes: {total_episodes} | Success Rate: {success_rate:.1f}%")
        
        # Final statistics
        print("\n" + "=" * 40)
        print("Final Statistics:")
        print(f"Total episodes completed: {total_episodes}")
        if total_episodes > 0:
            success_rate = success_count / total_episodes * 100
            print(f"Success rate: {success_rate:.1f}% ({success_count}/{total_episodes})")
            
            if hasattr(env, 'get_success_rate'):
                current_success_rate = env.get_success_rate() * 100
                print(f"Current success rate: {current_success_rate:.1f}%")
        
        # Close environment
        env.close()
        print("\nEnvironment closed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Check if Isaac Gym is installed
        try:
            import isaacgym
            print("Isaac Gym is installed.")
        except ImportError:
            print("\nIsaac Gym is not installed!")
            print("Please install Isaac Gym:")
            print("1. Download Isaac Gym from NVIDIA Developer website")
            print("2. Follow the installation instructions")
            print("3. Install the Python package: pip install -e isaacgym/python")
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            print("\nCUDA is not available!")
            print("Isaac Gym requires CUDA for GPU acceleration.")
            print("Consider setting device='cpu' for CPU-only operation (slower).")
        
        return 1
    
    return 0


def demo_multi_environment():
    """Demonstrate multi-environment capabilities."""
    print("\n" + "=" * 60)
    print("Multi-Environment Demo")
    print("=" * 60)
    
    # Configuration for multiple environments
    config = {
        "obs_type": "state",
        "render_mode": "rgb_array",  # Use rgb_array for faster operation
        "num_envs": 16,  # More environments for better parallelization
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "headless": True,  # Headless for better performance
    }
    
    try:
        env = LiftIsaac(**config)
        
        print(f"Created {config['num_envs']} parallel environments")
        print(f"Using device: {config['device']}")
        
        # Reset all environments
        obs, info = env.reset()
        
        # Run for a shorter time with more environments
        total_steps = 500
        step_rewards = []
        
        for step in range(total_steps):
            # Random actions for all environments
            actions = np.random.uniform(-1, 1, (config["num_envs"], env.action_space.shape[0]))
            
            # Step all environments simultaneously
            obs, rewards, terminated, truncated, info = env.step(actions)
            step_rewards.append(rewards.mean())
            
            if (step + 1) % 100 == 0:
                avg_reward = np.mean(step_rewards[-100:])
                print(f"Step {step + 1}/{total_steps} | Avg Reward: {avg_reward:.3f}")
        
        print(f"\nDemo completed with {config['num_envs']} parallel environments")
        env.close()
        
    except Exception as e:
        print(f"Multi-environment demo failed: {e}")


if __name__ == "__main__":
    # Run main demo
    result = main()
    
    # Run multi-environment demo if main demo succeeded
    if result == 0:
        demo_multi_environment()
    
    print("\nDemo completed!") 