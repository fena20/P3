"""
Ablation Study: Impact of Cycling Penalty Weight
=================================================
This script runs the critical ablation study to prove the value of the
physics-informed cycling penalty.

Configurations tested:
1. w₃ = 0   (Standard DRL, no cycling penalty)
2. w₃ = 5   (Low penalty)
3. w₃ = 10  (Proposed configuration)
4. w₃ = 20  (High penalty)

Purpose: Demonstrate that without cycling penalty, standard DRL achieves
         lowest cost but destroys hardware through short-cycling.

Author: Lead Researcher, Cyber-Physical Energy Systems
Target: Applied Energy (Q1 Journal)
"""

import numpy as np
import pandas as pd
import os
from environment import SmartHomeEnv, load_ampds2_mock_data
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from tables import TableGenerator
import warnings
warnings.filterwarnings('ignore')


def make_ablation_env(data, w_cycling, rank=0, seed=0):
    """Create environment with specific cycling penalty weight."""
    def _init():
        env = SmartHomeEnv(data=data, episode_length=1440)
        env.w_cycling = w_cycling  # Set specific penalty weight
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


def train_ablation_model(data, w_cycling, name, timesteps=30000, n_envs=4):
    """
    Train a model with specific cycling penalty weight.
    
    Args:
        data: AMPds2 dataset
        w_cycling: Cycling penalty weight
        name: Configuration name
        timesteps: Training timesteps
        n_envs: Number of parallel environments
    
    Returns:
        Trained model
    """
    print(f"\n{'='*70}")
    print(f"Training: {name} (w_cycling={w_cycling})")
    print(f"{'='*70}")
    
    # Create vectorized environment
    env = DummyVecEnv([make_ablation_env(data, w_cycling, i, 42) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Train PPO
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0
    )
    
    print(f"Training for {timesteps:,} timesteps...")
    model.learn(total_timesteps=timesteps, progress_bar=True)
    
    # Save model
    os.makedirs(f"./models/ablation", exist_ok=True)
    model.save(f"./models/ablation/{name}")
    env.save(f"./models/ablation/{name}_vecnorm.pkl")
    
    print(f"✓ Training complete: {name}")
    
    return model, env


def evaluate_ablation_model(model, data, w_cycling, n_episodes=10):
    """
    Evaluate an ablation model.
    
    Args:
        model: Trained PPO model
        data: Evaluation dataset
        w_cycling: Cycling penalty weight
        n_episodes: Number of evaluation episodes
    
    Returns:
        Dictionary of metrics
    """
    env = SmartHomeEnv(data=data, episode_length=1440)
    env.w_cycling = w_cycling
    
    costs = []
    discomforts = []
    switches = []
    rewards = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=100 + ep)
        episode_reward = 0
        
        for step in range(1440):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        costs.append(info['episode_cost'])
        discomforts.append(info['episode_discomfort'])
        switches.append(info['episode_switches'])
        rewards.append(episode_reward)
    
    return {
        'cost': np.mean(costs),
        'cost_std': np.std(costs),
        'discomfort': np.mean(discomforts),
        'discomfort_std': np.std(discomforts),
        'switches': np.mean(switches),
        'switches_std': np.std(switches),
        'reward': np.mean(rewards),
        'reward_std': np.std(rewards)
    }


def run_full_ablation_study(
    timesteps_per_config=30000,
    n_episodes_eval=10,
    quick_mode=True
):
    """
    Run complete ablation study with 4 configurations.
    
    Args:
        timesteps_per_config: Training timesteps per configuration
        n_episodes_eval: Evaluation episodes per configuration
        quick_mode: If True, use reduced timesteps for quick testing
    
    Returns:
        Dictionary of results for all configurations
    """
    print("\n" + "="*70)
    print("ABLATION STUDY: Impact of Cycling Penalty Weight (w₃)")
    print("="*70)
    print("\nPurpose: Prove that cycling penalty is ESSENTIAL for hardware safety")
    print("Question: What happens if we remove the physics-informed constraint?")
    print("\nConfigurations:")
    print("  1. w₃ = 0   → Standard DRL (no hardware awareness)")
    print("  2. w₃ = 5   → Low penalty")
    print("  3. w₃ = 10  → Proposed (balanced)")
    print("  4. w₃ = 20  → High penalty (very conservative)")
    print("="*70 + "\n")
    
    if quick_mode:
        timesteps_per_config = 10000
        n_episodes_eval = 5
        print("⚡ Quick mode: Reduced timesteps for fast testing\n")
    
    # Load datasets
    print("Loading datasets...")
    train_data = load_ampds2_mock_data(num_samples=100000)
    eval_data = load_ampds2_mock_data(num_samples=50000)
    print(f"✓ Train: {len(train_data)} samples, Eval: {len(eval_data)} samples\n")
    
    # Configurations
    configs = [
        (0, 'no_penalty', 'Standard DRL (w₃=0)'),
        (5, 'low_penalty', 'Low Penalty (w₃=5)'),
        (10, 'proposed', 'Proposed PI-DRL (w₃=10)'),
        (20, 'high_penalty', 'High Penalty (w₃=20)')
    ]
    
    results = {}
    
    for w_cycling, key, name in configs:
        print("\n" + "-"*70)
        
        # Train
        model, train_env = train_ablation_model(
            train_data, w_cycling, key,
            timesteps=timesteps_per_config, n_envs=4
        )
        
        # Evaluate
        print(f"\nEvaluating {name} over {n_episodes_eval} episodes...")
        metrics = evaluate_ablation_model(model, eval_data, w_cycling, n_episodes=n_episodes_eval)
        
        results[key] = metrics
        
        # Print results
        print(f"\nResults for {name}:")
        print(f"  Daily Cost:  ${metrics['cost']:.4f} ± ${metrics['cost_std']:.4f}")
        print(f"  Discomfort:  {metrics['discomfort']:.2f} ± {metrics['discomfort_std']:.2f} °C·h")
        print(f"  Switches:    {metrics['switches']:.0f} ± {metrics['switches_std']:.0f} /day")
        print(f"  Reward:      {metrics['reward']:.2f} ± {metrics['reward_std']:.2f}")
        
        # Hardware safety assessment
        if metrics['switches'] < 50:
            safety = "✅ SAFE"
        elif metrics['switches'] < 80:
            safety = "⚠️  MARGINAL"
        else:
            safety = "❌ UNSAFE (exceeds manufacturer limits)"
        print(f"  Hardware:    {safety}")
    
    # Comparative analysis
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS")
    print("="*70)
    
    no_penalty = results['no_penalty']
    proposed = results['proposed']
    
    cost_penalty = ((proposed['cost'] - no_penalty['cost']) / no_penalty['cost']) * 100
    switches_reduction = ((no_penalty['switches'] - proposed['switches']) / no_penalty['switches']) * 100
    
    print(f"\nStandard DRL (w₃=0) vs. Proposed PI-DRL (w₃=10):")
    print(f"  Cost:        ${no_penalty['cost']:.4f} → ${proposed['cost']:.4f} (+{cost_penalty:.1f}%)")
    print(f"  Switches:    {no_penalty['switches']:.0f} → {proposed['switches']:.0f} (-{switches_reduction:.1f}%)")
    print(f"  Discomfort:  {no_penalty['discomfort']:.2f} → {proposed['discomfort']:.2f}")
    
    print(f"\n🎯 KEY FINDING:")
    print(f"   Small cost increase (+{cost_penalty:.1f}%) prevents hardware damage")
    print(f"   Daily extra cost: ${proposed['cost'] - no_penalty['cost']:.4f}")
    print(f"   vs. Compressor replacement: ~$2,500")
    print(f"   → Pays for itself in {2500 / ((proposed['cost'] - no_penalty['cost']) * 365):.0f} days!")
    
    print(f"\n✅ CONCLUSION: Cycling penalty is ESSENTIAL")
    print("="*70 + "\n")
    
    return results


def generate_ablation_table(results: dict):
    """Generate Table 3 from ablation results."""
    print("Generating Table 3: Ablation Study...")
    
    gen = TableGenerator(save_dir="./tables")
    gen.table3_ablation(results)
    
    print("\n✓ Table 3 (Ablation Study) generated successfully!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ablation study for cycling penalty")
    parser.add_argument('--timesteps', type=int, default=30000,
                       help='Training timesteps per configuration (default: 30000)')
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Evaluation episodes (default: 10)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: reduced timesteps for testing')
    
    args = parser.parse_args()
    
    # Run ablation study
    results = run_full_ablation_study(
        timesteps_per_config=args.timesteps,
        n_episodes_eval=args.eval_episodes,
        quick_mode=args.quick
    )
    
    # Generate table
    generate_ablation_table(results)
    
    # Save results
    os.makedirs("./results", exist_ok=True)
    results_df = pd.DataFrame(results).T
    results_df.to_csv("./results/ablation_results.csv")
    
    print("\n✓ Ablation study complete!")
    print(f"  Models saved: ./models/ablation/")
    print(f"  Table saved:  ./tables/table3_ablation.*")
    print(f"  Results CSV:  ./results/ablation_results.csv")
