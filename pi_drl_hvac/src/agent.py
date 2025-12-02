"""
Physics-Informed PPO Agent for HVAC Control
============================================

This module implements the training pipeline for our PI-DRL framework
using Proximal Policy Optimization (PPO) from Stable-Baselines3.

Key Features:
1. Custom callback for model checkpointing
2. Training with physics-informed environment
3. Model loading and evaluation utilities

PPO Reference:
- Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms.
  arXiv preprint arXiv:1707.06347.

Author: CPES Research Lab
Target: Applied Energy (Q1 Journal)
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, 
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

from .environment import SmartHomeEnv, BaselineThemostatEnv
from .data_loader import SyntheticDataGenerator


class BestModelCallback(BaseCallback):
    """
    Custom callback for saving the best model during training.
    
    This callback:
    1. Evaluates the agent periodically
    2. Saves the model when performance improves
    3. Logs training metrics for TensorBoard
    
    Attributes:
        eval_env: Environment for evaluation
        best_mean_reward: Best mean reward achieved
        save_path: Path to save the best model
    """
    
    def __init__(
        self,
        eval_env: SmartHomeEnv,
        save_path: str,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5,
        verbose: int = 1
    ):
        """
        Initialize the callback.
        
        Args:
            eval_env: Environment for evaluation
            save_path: Directory to save models
            eval_freq: Evaluation frequency (steps)
            n_eval_episodes: Number of episodes per evaluation
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        
        self.best_mean_reward = -np.inf
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        
    def _on_step(self) -> bool:
        """
        Called at each training step.
        
        Returns:
            continue_training: Whether to continue training
        """
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current policy
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                return_episode_rewards=False
            )
            
            self.evaluations_results.append(mean_reward)
            self.evaluations_timesteps.append(self.num_timesteps)
            
            if self.verbose > 0:
                print(f"\nStep {self.num_timesteps}: "
                      f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.save_path / "best_model")
                if self.verbose > 0:
                    print(f"  → New best model saved! (reward: {mean_reward:.2f})")
            
            # Log to TensorBoard
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/best_reward", self.best_mean_reward)
            
        return True
    
    def _on_training_end(self):
        """Called at the end of training."""
        # Save final model
        self.model.save(self.save_path / "final_model")
        
        # Save evaluation history
        eval_df = pd.DataFrame({
            'timestep': self.evaluations_timesteps,
            'mean_reward': self.evaluations_results
        })
        eval_df.to_csv(self.save_path / "evaluation_history.csv", index=False)


class CyclingMonitorCallback(BaseCallback):
    """
    Callback to monitor cycling behavior during training.
    
    Tracks:
    - Number of cycles per episode
    - Time between state changes
    - Short-cycling events (switches < 15 min apart)
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_cycles = []
        self.short_cycling_events = []
        
    def _on_step(self) -> bool:
        """Log cycling metrics."""
        # Access info from environment
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'cycle_count' in info:
                self.logger.record("train/cycles", info['cycle_count'])
            if 'time_since_switch' in info:
                self.logger.record("train/time_since_switch", 
                                   info['time_since_switch'])
        return True


class PI_DRL_Agent:
    """
    Physics-Informed Deep Reinforcement Learning Agent.
    
    This class wraps the PPO agent and provides:
    1. Training with custom callbacks
    2. Evaluation against baseline
    3. Policy analysis utilities
    
    Hyperparameters are tuned for the HVAC control task:
    - Moderate learning rate for stable learning
    - Larger batch size for variance reduction
    - GAE lambda for advantage estimation
    
    Attributes:
        env: Training environment
        model: PPO model from Stable-Baselines3
        save_dir: Directory for saving models/logs
    """
    
    # Optimized hyperparameters for HVAC control
    DEFAULT_HYPERPARAMS = {
        'learning_rate': 3e-4,
        'n_steps': 2048,           # Steps per update
        'batch_size': 64,          # Minibatch size
        'n_epochs': 10,            # Epochs per update
        'gamma': 0.99,             # Discount factor
        'gae_lambda': 0.95,        # GAE parameter
        'clip_range': 0.2,         # PPO clipping
        'ent_coef': 0.01,          # Entropy bonus (exploration)
        'vf_coef': 0.5,            # Value function coefficient
        'max_grad_norm': 0.5,      # Gradient clipping
        'verbose': 1
    }
    
    def __init__(
        self,
        env: SmartHomeEnv,
        save_dir: str = "models",
        hyperparams: Optional[Dict] = None,
        seed: int = 42
    ):
        """
        Initialize the PI-DRL agent.
        
        Args:
            env: SmartHomeEnv instance for training
            save_dir: Directory for saving models and logs
            hyperparams: Custom hyperparameters (merged with defaults)
            seed: Random seed for reproducibility
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup environment with monitoring
        self.env = Monitor(env)
        
        # Merge hyperparameters
        self.hyperparams = self.DEFAULT_HYPERPARAMS.copy()
        if hyperparams:
            self.hyperparams.update(hyperparams)
        
        # Initialize PPO model
        self.model = PPO(
            "MlpPolicy",
            self.env,
            seed=seed,
            tensorboard_log=str(self.save_dir / "logs"),
            **self.hyperparams
        )
        
        self.training_history = {
            'rewards': [],
            'episode_lengths': [],
            'value_losses': [],
            'policy_losses': []
        }
        
    def train(
        self,
        total_timesteps: int = 100000,
        eval_env: Optional[SmartHomeEnv] = None,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5,
        save_best: bool = True
    ) -> Dict[str, Any]:
        """
        Train the PPO agent.
        
        Args:
            total_timesteps: Total training steps
            eval_env: Environment for evaluation (creates new if None)
            eval_freq: Evaluation frequency
            n_eval_episodes: Episodes per evaluation
            save_best: Whether to save the best model
            
        Returns:
            training_info: Dictionary with training statistics
        """
        print("=" * 60)
        print("PHYSICS-INFORMED DRL TRAINING")
        print("=" * 60)
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Save directory: {self.save_dir}")
        print(f"Hyperparameters: {self.hyperparams}")
        print("=" * 60)
        
        # Setup evaluation environment
        if eval_env is None:
            eval_env = SmartHomeEnv(episode_length=1440, random_seed=123)
        
        # Setup callbacks
        callbacks = []
        
        if save_best:
            best_callback = BestModelCallback(
                eval_env=eval_env,
                save_path=str(self.save_dir),
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes
            )
            callbacks.append(best_callback)
        
        # Add cycling monitor
        callbacks.append(CyclingMonitorCallback())
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=str(self.save_dir / "checkpoints"),
            name_prefix="ppo_hvac"
        )
        callbacks.append(checkpoint_callback)
        
        callback_list = CallbackList(callbacks)
        
        # Train
        start_time = datetime.now()
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True
        )
        training_time = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print(f"Time elapsed: {training_time:.1f} seconds")
        print("=" * 60)
        
        return {
            'training_time': training_time,
            'best_reward': callbacks[0].best_mean_reward if save_best else None,
            'final_model_path': str(self.save_dir / "final_model.zip")
        }
    
    def load(self, model_path: str):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        self.model = PPO.load(model_path, env=self.env)
        print(f"Model loaded from {model_path}")
    
    def evaluate(
        self,
        env: Optional[SmartHomeEnv] = None,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Tuple[float, float, pd.DataFrame]:
        """
        Evaluate the trained agent.
        
        Args:
            env: Evaluation environment (uses training env if None)
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic actions
            
        Returns:
            mean_reward: Mean episode reward
            std_reward: Standard deviation of rewards
            results_df: DataFrame with episode-level metrics
        """
        if env is None:
            env = self.env
        
        episode_rewards = []
        episode_lengths = []
        episode_costs = []
        episode_cycles = []
        episode_comfort_violations = []
        
        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            ep_reward = 0
            step = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                step += 1
                done = terminated or truncated
            
            episode_rewards.append(ep_reward)
            episode_lengths.append(step)
            episode_costs.append(info.get('total_cost', 0))
            episode_cycles.append(info.get('cycle_count', 0))
            episode_comfort_violations.append(info.get('comfort_violations', 0))
        
        results_df = pd.DataFrame({
            'episode': range(n_episodes),
            'reward': episode_rewards,
            'length': episode_lengths,
            'cost': episode_costs,
            'cycles': episode_cycles,
            'comfort_violations': episode_comfort_violations
        })
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        print(f"\nEvaluation over {n_episodes} episodes:")
        print(f"  Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"  Mean cost: ${np.mean(episode_costs):.4f}")
        print(f"  Mean cycles: {np.mean(episode_cycles):.1f}")
        print(f"  Mean comfort violations: {np.mean(episode_comfort_violations):.1f}")
        
        return mean_reward, std_reward, results_df
    
    def get_action_probabilities(
        self,
        observations: np.ndarray
    ) -> np.ndarray:
        """
        Get action probabilities for given observations.
        
        Useful for policy analysis and heatmap generation.
        
        Args:
            observations: Array of observations [N, obs_dim]
            
        Returns:
            probs: Probability of action=ON for each observation
        """
        import torch
        
        # Ensure correct shape
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)
        
        # Get policy distribution
        obs_tensor = torch.FloatTensor(observations)
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.numpy()
        
        # Return probability of action=ON (index 1)
        return probs[:, 1] if probs.ndim > 1 else probs[1]
    
    def run_comparison_episode(
        self,
        n_steps: int = 120,  # 2 hours at 1-min resolution
        start_index: int = 0
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run comparison between PI-DRL and baseline thermostat.
        
        Args:
            n_steps: Number of steps to simulate
            start_index: Starting index in data
            
        Returns:
            agent_history: DataFrame with PI-DRL agent trajectory
            baseline_history: DataFrame with baseline thermostat trajectory
        """
        # Create fresh environments with same initial conditions
        data_gen = SyntheticDataGenerator(random_seed=42)
        data = data_gen.generate(start_date="2012-04-01", end_date="2012-04-07")
        
        # PI-DRL Agent environment
        agent_env = SmartHomeEnv(
            data=data.copy(),
            episode_length=n_steps,
            random_seed=42
        )
        
        # Baseline thermostat environment
        baseline_env = BaselineThemostatEnv(
            data=data.copy(),
            episode_length=n_steps,
            random_seed=42
        )
        
        # Set same starting conditions
        options = {'start_index': start_index, 'initial_temp': 20.0}
        
        # Run PI-DRL agent
        obs, _ = agent_env.reset(options=options)
        for _ in range(n_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = agent_env.step(int(action))
            if term or trunc:
                break
        agent_history = agent_env.get_history_df()
        
        # Run baseline thermostat
        obs, _ = baseline_env.reset(options=options)
        for _ in range(n_steps):
            action = baseline_env.get_thermostat_action()
            obs, _, term, trunc, _ = baseline_env.step(action)
            if term or trunc:
                break
        baseline_history = baseline_env.get_history_df()
        
        return agent_history, baseline_history


def create_and_train_agent(
    total_timesteps: int = 50000,
    save_dir: str = "models",
    seed: int = 42
) -> PI_DRL_Agent:
    """
    Convenience function to create and train a PI-DRL agent.
    
    Args:
        total_timesteps: Total training steps
        save_dir: Directory for saving models
        seed: Random seed
        
    Returns:
        agent: Trained PI_DRL_Agent instance
    """
    # Create environment
    env = SmartHomeEnv(episode_length=1440, random_seed=seed)
    
    # Create agent
    agent = PI_DRL_Agent(env, save_dir=save_dir, seed=seed)
    
    # Train
    agent.train(total_timesteps=total_timesteps)
    
    return agent


if __name__ == "__main__":
    # Quick test of agent training
    print("Testing PI-DRL Agent...")
    
    # Create a small environment for testing
    env = SmartHomeEnv(episode_length=500, random_seed=42)
    
    # Create agent
    agent = PI_DRL_Agent(env, save_dir="test_models", seed=42)
    
    # Short training run
    agent.train(total_timesteps=5000, eval_freq=2500)
    
    # Evaluate
    mean_reward, std_reward, results = agent.evaluate(n_episodes=3)
    print(f"\nTest evaluation: {mean_reward:.2f} +/- {std_reward:.2f}")
