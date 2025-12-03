"""
PPO Training Script for Physics-Informed DRL Building Energy Management
========================================================================
This module implements PPO training with custom callbacks for model checkpointing
and performance monitoring.

Author: Lead Researcher, Cyber-Physical Energy Systems
Target: Applied Energy (Q1 Journal)
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

from environment import SmartHomeEnv, load_ampds2_mock_data


class MetricsCallback(BaseCallback):
    """
    Custom callback for tracking and logging training metrics.
    
    This callback captures episode-level statistics crucial for energy
    management research: cost, comfort violations, and equipment cycling.
    """
    
    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.episode_costs = []
        self.episode_discomforts = []
        self.episode_switches = []
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        """Called after every step in the environment."""
        # Check if episode ended
        for idx, done in enumerate(self.locals.get('dones', [])):
            if done:
                # Extract episode info
                info = self.locals['infos'][idx]
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                
                # Extract custom metrics
                if 'episode_cost' in info:
                    self.episode_costs.append(info['episode_cost'])
                if 'episode_discomfort' in info:
                    self.episode_discomforts.append(info['episode_discomfort'])
                if 'episode_switches' in info:
                    self.episode_switches.append(info['episode_switches'])
                
                # Log every 10 episodes
                if len(self.episode_rewards) % 10 == 0 and self.verbose > 0:
                    print(f"\n{'='*70}")
                    print(f"Episode {len(self.episode_rewards)}")
                    print(f"{'='*70}")
                    if len(self.episode_rewards) >= 10:
                        recent_rewards = self.episode_rewards[-10:]
                        recent_costs = self.episode_costs[-10:] if self.episode_costs else [0]
                        recent_switches = self.episode_switches[-10:] if self.episode_switches else [0]
                        
                        print(f"Mean Reward (last 10): {np.mean(recent_rewards):.2f}")
                        print(f"Mean Cost (last 10): ${np.mean(recent_costs):.4f}")
                        print(f"Mean Switches (last 10): {np.mean(recent_switches):.1f}")
                        print(f"{'='*70}\n")
        
        return True
    
    def get_metrics(self) -> dict:
        """Return collected metrics for post-training analysis."""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_costs': self.episode_costs,
            'episode_discomforts': self.episode_discomforts,
            'episode_switches': self.episode_switches
        }


def make_env(data: pd.DataFrame, rank: int = 0, seed: int = 0):
    """
    Create and wrap the environment with Monitor.
    
    Args:
        data: AMPds2 dataset
        rank: Index of the environment (for parallel envs)
        seed: Random seed
    
    Returns:
        Wrapped environment
    """
    def _init():
        env = SmartHomeEnv(data=data)
        env.reset(seed=seed + rank)
        # Monitor wrapper for automatic episode tracking
        env = Monitor(env)
        return env
    return _init


def train_ppo_agent(
    total_timesteps: int = 100000,
    n_envs: int = 4,
    save_dir: str = "./models",
    log_dir: str = "./logs",
    use_pretrained: bool = False,
    pretrained_path: str = None
):
    """
    Train a PPO agent on the Smart Home environment.
    
    Args:
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        save_dir: Directory to save model checkpoints
        log_dir: Directory to save training logs
        use_pretrained: Whether to load a pretrained model
        pretrained_path: Path to pretrained model
    
    Returns:
        Trained PPO model, training metrics, environment
    """
    print("="*70)
    print("Physics-Informed Deep Reinforcement Learning")
    print("Building Energy Management System")
    print("="*70)
    print(f"\nTraining Configuration:")
    print(f"  Total Timesteps: {total_timesteps:,}")
    print(f"  Parallel Envs: {n_envs}")
    print(f"  Algorithm: PPO (Proximal Policy Optimization)")
    print(f"  Save Directory: {save_dir}")
    print("="*70 + "\n")
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Load dataset
    print("Loading AMPds2 dataset (synthetic)...")
    data = load_ampds2_mock_data(num_samples=100000)  # ~70 days
    print(f"Dataset loaded: {len(data)} samples (~{len(data)/1440:.1f} days)\n")
    
    # Create vectorized environment
    print("Creating vectorized environment...")
    env = DummyVecEnv([make_env(data, i, 42) for i in range(n_envs)])
    
    # Normalize observations for better learning
    # This is crucial for environments with mixed scales (temp, solar, price)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    print(f"Environment created: {n_envs} parallel instances\n")
    
    # Initialize or load PPO model
    if use_pretrained and pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}...")
        model = PPO.load(pretrained_path, env=env)
        print("Pretrained model loaded!\n")
    else:
        print("Initializing new PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,  # Number of steps to run per update
            batch_size=64,
            n_epochs=10,
            gamma=0.99,  # Discount factor
            gae_lambda=0.95,  # GAE lambda for advantage estimation
            clip_range=0.2,  # PPO clipping parameter
            ent_coef=0.01,  # Entropy coefficient for exploration
            vf_coef=0.5,  # Value function coefficient
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=log_dir
        )
        print("PPO model initialized with hyperparameters:")
        print(f"  Learning Rate: 3e-4")
        print(f"  Steps per Update: 2048")
        print(f"  Batch Size: 64")
        print(f"  Entropy Coefficient: 0.01 (exploration)\n")
    
    # Setup callbacks
    print("Setting up training callbacks...")
    
    # Checkpoint callback - save every 10k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000 // n_envs,
        save_path=save_dir,
        name_prefix="ppo_smarthome",
        save_replay_buffer=False,
        save_vecnormalize=True
    )
    
    # Metrics callback - track custom metrics
    metrics_callback = MetricsCallback(verbose=1)
    
    # Combine callbacks
    callbacks = [checkpoint_callback, metrics_callback]
    print("Callbacks configured: Checkpoint + Metrics\n")
    
    # Train the agent
    print("="*70)
    print("Starting Training...")
    print("="*70)
    print("(This may take several minutes...)\n")
    
    start_time = datetime.now()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    training_time = datetime.now() - start_time
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Training Time: {training_time}")
    print(f"Final Model Saved: {save_dir}/ppo_smarthome_final\n")
    
    # Save final model
    final_model_path = os.path.join(save_dir, "ppo_smarthome_final")
    model.save(final_model_path)
    env.save(os.path.join(save_dir, "vec_normalize.pkl"))
    
    # Get training metrics
    metrics = metrics_callback.get_metrics()
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'episode': range(len(metrics['episode_rewards'])),
        'reward': metrics['episode_rewards'],
        'cost': metrics['episode_costs'] if metrics['episode_costs'] else [0] * len(metrics['episode_rewards']),
        'discomfort': metrics['episode_discomforts'] if metrics['episode_discomforts'] else [0] * len(metrics['episode_rewards']),
        'switches': metrics['episode_switches'] if metrics['episode_switches'] else [0] * len(metrics['episode_rewards'])
    })
    metrics_path = os.path.join(log_dir, "training_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Training metrics saved: {metrics_path}\n")
    
    return model, metrics, env


def evaluate_agent(model, env, n_episodes: int = 10):
    """
    Evaluate the trained agent over multiple episodes.
    
    Args:
        model: Trained PPO model
        env: Environment (can be vectorized)
        n_episodes: Number of evaluation episodes
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("="*70)
    print(f"Evaluating Agent over {n_episodes} episodes...")
    print("="*70 + "\n")
    
    episode_rewards = []
    episode_costs = []
    episode_discomforts = []
    episode_switches = []
    
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_cost = 0
        ep_discomfort = 0
        ep_switches = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            
            # Extract info (handle vectorized env)
            if isinstance(info, list):
                info = info[0]
            
            if 'cost' in info:
                ep_cost += info['cost']
            if 'discomfort' in info:
                ep_discomfort += info['discomfort']
            if 'episode_switches' in info:
                ep_switches = info['episode_switches']
            
            if isinstance(done, np.ndarray):
                done = done[0]
        
        episode_rewards.append(ep_reward)
        episode_costs.append(ep_cost)
        episode_discomforts.append(ep_discomfort)
        episode_switches.append(ep_switches)
        
        print(f"Episode {ep+1}/{n_episodes}: Reward={ep_reward:.2f}, "
              f"Cost=${ep_cost:.4f}, Switches={ep_switches}")
    
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_cost': np.mean(episode_costs),
        'std_cost': np.std(episode_costs),
        'mean_discomfort': np.mean(episode_discomforts),
        'std_discomfort': np.std(episode_discomforts),
        'mean_switches': np.mean(episode_switches),
        'std_switches': np.std(episode_switches)
    }
    
    print("\n" + "="*70)
    print("Evaluation Results:")
    print("="*70)
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Cost: ${results['mean_cost']:.4f} ± ${results['std_cost']:.4f}")
    print(f"Mean Discomfort: {results['mean_discomfort']:.2f} ± {results['std_discomfort']:.2f}")
    print(f"Mean Switches: {results['mean_switches']:.1f} ± {results['std_switches']:.1f}")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    # Train the PPO agent
    model, metrics, env = train_ppo_agent(
        total_timesteps=50000,  # Reduced for quick demo
        n_envs=4,
        save_dir="./models",
        log_dir="./logs"
    )
    
    # Evaluate the trained agent
    eval_results = evaluate_agent(model, env, n_episodes=5)
    
    print("\nTraining script complete! Ready for visualization.")
