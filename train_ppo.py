"""
PPO Training Script for Physics-Informed Smart Home Control
Uses stable-baselines3 with callbacks for model checkpointing.
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from smarthome_env import SmartHomeEnv


class SaveBestModelCallback(BaseCallback):
    """
    Callback to save the best model based on evaluation reward.
    """
    def __init__(self, check_freq: int, save_path: str, verbose=1):
        super(SaveBestModelCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        
    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Evaluate current model
            mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            
            if self.verbose > 0:
                print(f"Step {self.n_calls}: Mean reward = {mean_reward:.2f}")
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"New best model! Saving to {self.save_path}")
                self.model.save(os.path.join(self.save_path, "best_model"))
        
        return True


def create_env():
    """Create and wrap the environment"""
    env = SmartHomeEnv()
    env = Monitor(env)
    return env


def train_ppo(total_timesteps=100000, 
              learning_rate=3e-4,
              n_steps=2048,
              batch_size=64,
              n_epochs=10,
              gamma=0.99,
              gae_lambda=0.95,
              clip_range=0.2,
              ent_coef=0.01,
              vf_coef=0.5,
              save_path="./models/ppo_smarthome"):
    """
    Train PPO agent on SmartHomeEnv
    
    Parameters:
    -----------
    total_timesteps : int
        Total number of timesteps to train
    learning_rate : float
        Learning rate for PPO
    n_steps : int
        Number of steps to collect per update
    batch_size : int
        Batch size for training
    n_epochs : int
        Number of epochs per update
    gamma : float
        Discount factor
    gae_lambda : float
        GAE lambda parameter
    clip_range : float
        PPO clip range
    ent_coef : float
        Entropy coefficient
    vf_coef : float
        Value function coefficient
    save_path : str
        Path to save models
    """
    
    # Create environment
    env = DummyVecEnv([create_env])
    
    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    
    # Create callbacks
    os.makedirs(save_path, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=save_path,
        name_prefix="ppo_smarthome"
    )
    
    save_best_callback = SaveBestModelCallback(
        check_freq=5000,
        save_path=save_path,
        verbose=1
    )
    
    # Train the agent
    print("Starting PPO training...")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Model will be saved to: {save_path}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, save_best_callback],
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(save_path, "final_model")
    model.save(final_model_path)
    print(f"\nTraining completed! Final model saved to: {final_model_path}")
    
    return model, env


if __name__ == "__main__":
    # Training configuration
    model, env = train_ppo(
        total_timesteps=50000,  # Adjust based on available time
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10
    )
    
    # Test the trained model
    print("\nTesting trained model...")
    obs = env.reset()
    total_reward = 0
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        if done[0]:
            break
    
    print(f"Test episode reward: {total_reward:.2f}")
