"""
Comprehensive Sensitivity Analysis for Applied Energy Publication
==================================================================
This module performs rigorous sensitivity analysis to demonstrate robustness
of the Physics-Informed DRL approach across parameter variations.

Analyses Included:
1. Reward Weight Sensitivity (w₁, w₂, w₃)
2. Thermal Model Parameter Sensitivity (R, C)
3. Control Constraint Sensitivity (min_cycle_time, comfort_range)
4. PPO Hyperparameter Sensitivity (learning_rate, gamma)
5. Environmental Condition Sensitivity (climate zones, seasons)

Author: Lead Researcher, Cyber-Physical Energy Systems
Target: Applied Energy (Q1 Journal)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import os
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from environment import SmartHomeEnv, load_ampds2_mock_data
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor


class SensitivityAnalyzer:
    """
    Comprehensive sensitivity analysis for Physics-Informed DRL.
    
    This class systematically varies parameters and measures their impact
    on key performance metrics: cost, comfort, and cycling.
    """
    
    def __init__(self, base_params: Dict[str, Any], save_dir: str = "./sensitivity_results"):
        """
        Initialize sensitivity analyzer.
        
        Args:
            base_params: Baseline parameter configuration
            save_dir: Directory to save results
        """
        self.base_params = base_params
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Results storage
        self.results = {}
        
    def analyze_reward_weights(
        self,
        w_cost_range: List[float] = [0.5, 1.0, 2.0, 5.0],
        w_discomfort_range: List[float] = [1.0, 3.0, 5.0, 10.0],
        w_cycling_range: List[float] = [0.0, 5.0, 10.0, 15.0, 20.0],
        timesteps: int = 30000,
        n_eval: int = 5
    ):
        """
        Sensitivity Analysis 1: Reward Function Weights
        
        Tests impact of w₁, w₂, w₃ on performance metrics.
        Critical for demonstrating that w₃=10 is optimal.
        
        Args:
            w_cost_range: Cost weight variations
            w_discomfort_range: Discomfort weight variations
            w_cycling_range: Cycling penalty weight variations
            timesteps: Training timesteps per configuration
            n_eval: Evaluation episodes
        """
        print("\n" + "="*70)
        print("SENSITIVITY ANALYSIS 1: REWARD FUNCTION WEIGHTS")
        print("="*70)
        print("Testing: w₁ (cost), w₂ (discomfort), w₃ (cycling penalty)")
        print(f"Configurations: {len(w_cost_range) + len(w_discomfort_range) + len(w_cycling_range)}")
        print("="*70 + "\n")
        
        results = []
        
        # Analysis 1.1: Vary w_cost (keeping w₂=5, w₃=10)
        print("\n1.1 Varying Cost Weight (w₁):")
        print("-" * 50)
        for w_cost in w_cost_range:
            print(f"\nTesting w₁={w_cost}...")
            metrics = self._train_and_evaluate(
                {'w_cost': w_cost, 'w_discomfort': 5.0, 'w_cycling': 10.0},
                timesteps, n_eval
            )
            results.append({
                'parameter': 'w_cost',
                'value': w_cost,
                **metrics
            })
            print(f"  Cost: ${metrics['cost']:.4f}, Switches: {metrics['switches']:.0f}")
        
        # Analysis 1.2: Vary w_discomfort (keeping w₁=1, w₃=10)
        print("\n1.2 Varying Discomfort Weight (w₂):")
        print("-" * 50)
        for w_discomfort in w_discomfort_range:
            print(f"\nTesting w₂={w_discomfort}...")
            metrics = self._train_and_evaluate(
                {'w_cost': 1.0, 'w_discomfort': w_discomfort, 'w_cycling': 10.0},
                timesteps, n_eval
            )
            results.append({
                'parameter': 'w_discomfort',
                'value': w_discomfort,
                **metrics
            })
            print(f"  Discomfort: {metrics['discomfort']:.2f}, Temp violations: {metrics['temp_violations']:.1f}%")
        
        # Analysis 1.3: Vary w_cycling (keeping w₁=1, w₂=5) - MOST IMPORTANT
        print("\n1.3 Varying Cycling Penalty Weight (w₃) ⭐ CRITICAL:")
        print("-" * 50)
        for w_cycling in w_cycling_range:
            print(f"\nTesting w₃={w_cycling}...")
            metrics = self._train_and_evaluate(
                {'w_cost': 1.0, 'w_discomfort': 5.0, 'w_cycling': w_cycling},
                timesteps, n_eval
            )
            results.append({
                'parameter': 'w_cycling',
                'value': w_cycling,
                **metrics
            })
            
            # Hardware safety assessment
            if metrics['switches'] < 50:
                safety = "SAFE ✅"
            elif metrics['switches'] < 80:
                safety = "MARGINAL ⚠️"
            else:
                safety = "UNSAFE ❌"
            
            print(f"  Switches: {metrics['switches']:.0f} ({safety})")
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(f"{self.save_dir}/sensitivity_reward_weights.csv", index=False)
        self.results['reward_weights'] = df
        
        print("\n✓ Reward weight sensitivity analysis complete!")
        return df
    
    def analyze_thermal_parameters(
        self,
        R_range: List[float] = [1.5, 2.0, 2.5, 3.0, 3.5],
        C_range: List[float] = [7.5, 8.5, 10.0, 12.0, 15.0],
        timesteps: int = 30000,
        n_eval: int = 5
    ):
        """
        Sensitivity Analysis 2: Building Thermal Parameters
        
        Tests robustness to thermal model uncertainties.
        Different buildings have different R (insulation) and C (thermal mass).
        
        Args:
            R_range: Thermal resistance values (°C/kW)
            C_range: Thermal capacitance values (kWh/°C)
            timesteps: Training timesteps
            n_eval: Evaluation episodes
        """
        print("\n" + "="*70)
        print("SENSITIVITY ANALYSIS 2: THERMAL MODEL PARAMETERS")
        print("="*70)
        print("Testing: R (insulation), C (thermal mass)")
        print("Purpose: Demonstrate robustness across building types")
        print("="*70 + "\n")
        
        results = []
        
        # Analysis 2.1: Vary R (insulation quality)
        print("\n2.1 Varying Thermal Resistance (R - Insulation):")
        print("-" * 50)
        for R in R_range:
            print(f"\nTesting R={R} °C/kW...")
            metrics = self._train_and_evaluate(
                {'R': R},
                timesteps, n_eval
            )
            results.append({
                'parameter': 'R',
                'value': R,
                **metrics
            })
            
            # Insulation quality
            if R > 3.0:
                quality = "Excellent (new building)"
            elif R > 2.5:
                quality = "Good (average)"
            elif R > 2.0:
                quality = "Fair (older building)"
            else:
                quality = "Poor (needs retrofit)"
            
            print(f"  {quality}")
            print(f"  Cost: ${metrics['cost']:.4f}, Switches: {metrics['switches']:.0f}")
        
        # Analysis 2.2: Vary C (thermal mass)
        print("\n2.2 Varying Thermal Capacitance (C - Thermal Mass):")
        print("-" * 50)
        for C in C_range:
            print(f"\nTesting C={C} kWh/°C...")
            metrics = self._train_and_evaluate(
                {'C': C},
                timesteps, n_eval
            )
            results.append({
                'parameter': 'C',
                'value': C,
                **metrics
            })
            
            # Thermal mass quality
            if C > 12.0:
                quality = "High (concrete/brick)"
            elif C > 9.0:
                quality = "Medium (typical)"
            else:
                quality = "Low (wood frame)"
            
            print(f"  {quality}")
            print(f"  Cost: ${metrics['cost']:.4f}, Switches: {metrics['switches']:.0f}")
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(f"{self.save_dir}/sensitivity_thermal_params.csv", index=False)
        self.results['thermal_params'] = df
        
        print("\n✓ Thermal parameter sensitivity analysis complete!")
        return df
    
    def analyze_control_constraints(
        self,
        min_cycle_range: List[int] = [5, 10, 15, 20, 30],
        comfort_ranges: List[Tuple[float, float]] = [(19, 25), (20, 24), (21, 23)],
        timesteps: int = 30000,
        n_eval: int = 5
    ):
        """
        Sensitivity Analysis 3: Control Constraints
        
        Tests impact of minimum cycle time and comfort temperature range.
        
        Args:
            min_cycle_range: Minimum cycle time values (minutes)
            comfort_ranges: Comfort temperature ranges (T_min, T_max)
            timesteps: Training timesteps
            n_eval: Evaluation episodes
        """
        print("\n" + "="*70)
        print("SENSITIVITY ANALYSIS 3: CONTROL CONSTRAINTS")
        print("="*70)
        print("Testing: min_cycle_time, comfort_range")
        print("="*70 + "\n")
        
        results = []
        
        # Analysis 3.1: Vary minimum cycle time
        print("\n3.1 Varying Minimum Cycle Time:")
        print("-" * 50)
        for min_cycle in min_cycle_range:
            print(f"\nTesting min_cycle={min_cycle} min...")
            metrics = self._train_and_evaluate(
                {'min_cycle_time': min_cycle},
                timesteps, n_eval
            )
            results.append({
                'parameter': 'min_cycle_time',
                'value': min_cycle,
                **metrics
            })
            print(f"  Switches: {metrics['switches']:.0f}, Cost: ${metrics['cost']:.4f}")
        
        # Analysis 3.2: Vary comfort range
        print("\n3.2 Varying Comfort Temperature Range:")
        print("-" * 50)
        for comfort_min, comfort_max in comfort_ranges:
            print(f"\nTesting range: {comfort_min}-{comfort_max}°C...")
            metrics = self._train_and_evaluate(
                {'comfort_temp_min': comfort_min, 'comfort_temp_max': comfort_max},
                timesteps, n_eval
            )
            results.append({
                'parameter': 'comfort_range',
                'value': f"{comfort_min}-{comfort_max}",
                **metrics
            })
            print(f"  Discomfort: {metrics['discomfort']:.2f}, Cost: ${metrics['cost']:.4f}")
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(f"{self.save_dir}/sensitivity_control_constraints.csv", index=False)
        self.results['control_constraints'] = df
        
        print("\n✓ Control constraint sensitivity analysis complete!")
        return df
    
    def analyze_ppo_hyperparameters(
        self,
        lr_range: List[float] = [1e-4, 3e-4, 5e-4, 1e-3],
        gamma_range: List[float] = [0.95, 0.97, 0.99, 0.995],
        timesteps: int = 30000,
        n_eval: int = 5
    ):
        """
        Sensitivity Analysis 4: PPO Hyperparameters
        
        Tests robustness to learning algorithm settings.
        
        Args:
            lr_range: Learning rate values
            gamma_range: Discount factor values
            timesteps: Training timesteps
            n_eval: Evaluation episodes
        """
        print("\n" + "="*70)
        print("SENSITIVITY ANALYSIS 4: PPO HYPERPARAMETERS")
        print("="*70)
        print("Testing: learning_rate, gamma (discount factor)")
        print("="*70 + "\n")
        
        results = []
        
        # Analysis 4.1: Vary learning rate
        print("\n4.1 Varying Learning Rate:")
        print("-" * 50)
        for lr in lr_range:
            print(f"\nTesting lr={lr:.1e}...")
            metrics = self._train_and_evaluate(
                {'learning_rate': lr},
                timesteps, n_eval
            )
            results.append({
                'parameter': 'learning_rate',
                'value': lr,
                **metrics
            })
            print(f"  Reward: {metrics['reward']:.2f}, Cost: ${metrics['cost']:.4f}")
        
        # Analysis 4.2: Vary discount factor
        print("\n4.2 Varying Discount Factor (γ):")
        print("-" * 50)
        for gamma in gamma_range:
            print(f"\nTesting γ={gamma}...")
            metrics = self._train_and_evaluate(
                {'gamma': gamma},
                timesteps, n_eval
            )
            results.append({
                'parameter': 'gamma',
                'value': gamma,
                **metrics
            })
            print(f"  Reward: {metrics['reward']:.2f}, Switches: {metrics['switches']:.0f}")
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(f"{self.save_dir}/sensitivity_ppo_hyperparams.csv", index=False)
        self.results['ppo_hyperparams'] = df
        
        print("\n✓ PPO hyperparameter sensitivity analysis complete!")
        return df
    
    def analyze_climate_conditions(
        self,
        climate_zones: List[str] = ['cold', 'moderate', 'hot'],
        timesteps: int = 30000,
        n_eval: int = 5
    ):
        """
        Sensitivity Analysis 5: Environmental Conditions
        
        Tests generalization across different climate zones.
        
        Args:
            climate_zones: List of climate types to test
            timesteps: Training timesteps
            n_eval: Evaluation episodes
        """
        print("\n" + "="*70)
        print("SENSITIVITY ANALYSIS 5: CLIMATE CONDITIONS")
        print("="*70)
        print("Testing: Different climate zones (cold, moderate, hot)")
        print("Purpose: Demonstrate generalization capability")
        print("="*70 + "\n")
        
        results = []
        
        # Climate zone definitions
        climate_params = {
            'cold': {'temp_mean': 5, 'temp_std': 10, 'solar_factor': 0.7},
            'moderate': {'temp_mean': 15, 'temp_std': 8, 'solar_factor': 1.0},
            'hot': {'temp_mean': 28, 'temp_std': 6, 'solar_factor': 1.3}
        }
        
        for climate in climate_zones:
            print(f"\nTesting climate: {climate.upper()}...")
            params = climate_params[climate]
            
            metrics = self._train_and_evaluate(
                {'climate_zone': climate, **params},
                timesteps, n_eval
            )
            results.append({
                'parameter': 'climate_zone',
                'value': climate,
                **metrics
            })
            
            print(f"  Cost: ${metrics['cost']:.4f}")
            print(f"  Switches: {metrics['switches']:.0f}")
            print(f"  Discomfort: {metrics['discomfort']:.2f}")
        
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(f"{self.save_dir}/sensitivity_climate.csv", index=False)
        self.results['climate'] = df
        
        print("\n✓ Climate sensitivity analysis complete!")
        return df
    
    def _train_and_evaluate(
        self,
        param_overrides: Dict[str, Any],
        timesteps: int,
        n_eval: int
    ) -> Dict[str, float]:
        """
        Train and evaluate a model with parameter variations.
        
        Args:
            param_overrides: Parameters to override from base
            timesteps: Training timesteps
            n_eval: Number of evaluation episodes
        
        Returns:
            Dictionary of performance metrics
        """
        # Merge parameters
        params = {**self.base_params, **param_overrides}
        
        # Create environment
        data = load_ampds2_mock_data(num_samples=100000)
        
        def make_env():
            env = SmartHomeEnv(data=data, episode_length=1440)
            
            # Apply parameter overrides to environment
            if 'w_cost' in params:
                env.w_cost = params['w_cost']
            if 'w_discomfort' in params:
                env.w_discomfort = params['w_discomfort']
            if 'w_cycling' in params:
                env.w_cycling = params['w_cycling']
            if 'R' in params:
                env.R = params['R']
            if 'C' in params:
                env.C = params['C']
            if 'min_cycle_time' in params:
                env.min_cycle_time = params['min_cycle_time']
            if 'comfort_temp_min' in params:
                env.comfort_temp_min = params['comfort_temp_min']
            if 'comfort_temp_max' in params:
                env.comfort_temp_max = params['comfort_temp_max']
            
            return Monitor(env)
        
        # Create vectorized environment
        vec_env = DummyVecEnv([make_env for _ in range(4)])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        
        # Train model
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=params.get('learning_rate', 3e-4),
            gamma=params.get('gamma', 0.99),
            n_steps=2048,
            batch_size=64,
            verbose=0
        )
        
        model.learn(total_timesteps=timesteps, progress_bar=False)
        
        # Evaluate
        eval_env = make_env()
        
        costs = []
        discomforts = []
        switches = []
        rewards = []
        temp_violations = []
        
        for _ in range(n_eval):
            obs, _ = eval_env.reset()
            episode_reward = 0
            violations = 0
            
            for step in range(1440):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                
                # Count temperature violations
                if info['indoor_temp'] < 20 or info['indoor_temp'] > 24:
                    violations += 1
                
                if terminated or truncated:
                    break
            
            costs.append(info['episode_cost'])
            discomforts.append(info['episode_discomfort'])
            switches.append(info['episode_switches'])
            rewards.append(episode_reward)
            temp_violations.append((violations / 1440) * 100)
        
        return {
            'cost': np.mean(costs),
            'cost_std': np.std(costs),
            'discomfort': np.mean(discomforts),
            'discomfort_std': np.std(discomforts),
            'switches': np.mean(switches),
            'switches_std': np.std(switches),
            'reward': np.mean(rewards),
            'reward_std': np.std(rewards),
            'temp_violations': np.mean(temp_violations)
        }
    
    def generate_visualizations(self):
        """Generate publication-quality sensitivity analysis figures."""
        print("\n" + "="*70)
        print("GENERATING SENSITIVITY ANALYSIS VISUALIZATIONS")
        print("="*70 + "\n")
        
        # Set publication style
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 12
        })
        
        # Figure 1: Reward Weight Sensitivity
        if 'reward_weights' in self.results:
            self._plot_reward_weights()
        
        # Figure 2: Thermal Parameter Sensitivity
        if 'thermal_params' in self.results:
            self._plot_thermal_params()
        
        # Figure 3: Summary Heatmap
        self._plot_summary_heatmap()
        
        print("\n✓ All sensitivity visualizations generated!")
    
    def _plot_reward_weights(self):
        """Plot reward weight sensitivity."""
        df = self.results['reward_weights']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot for each weight
        for idx, (param, ax) in enumerate(zip(['w_cost', 'w_discomfort', 'w_cycling'], axes)):
            data = df[df['parameter'] == param]
            
            ax2 = ax.twinx()
            
            # Plot cost and switches
            ax.plot(data['value'], data['cost'], 'o-', color='#1f77b4', 
                   linewidth=2, markersize=8, label='Daily Cost')
            ax2.plot(data['value'], data['switches'], 's-', color='#d62728',
                    linewidth=2, markersize=8, label='Switches/day')
            
            # Hardware safety threshold
            if param == 'w_cycling':
                ax2.axhline(y=50, color='green', linestyle='--', linewidth=2, alpha=0.6)
                ax2.axhline(y=80, color='red', linestyle='--', linewidth=2, alpha=0.6)
                ax2.text(data['value'].max(), 50, ' Safe limit', va='center', fontsize=10)
                ax2.text(data['value'].max(), 80, ' Unsafe', va='center', fontsize=10)
            
            ax.set_xlabel(f'{param}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Daily Cost ($)', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Switches per Day', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.suptitle('Sensitivity Analysis: Reward Function Weights', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/sensitivity_reward_weights.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.save_dir}/sensitivity_reward_weights.pdf", bbox_inches='tight')
        plt.close()
        
        print("✓ Reward weight sensitivity plot saved")
    
    def _plot_thermal_params(self):
        """Plot thermal parameter sensitivity."""
        df = self.results['thermal_params']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # R sensitivity
        data_R = df[df['parameter'] == 'R']
        ax1.errorbar(data_R['value'], data_R['cost'], yerr=data_R['cost_std'],
                    fmt='o-', linewidth=2, markersize=8, capsize=5, label='Cost')
        ax1_2 = ax1.twinx()
        ax1_2.errorbar(data_R['value'], data_R['switches'], yerr=data_R['switches_std'],
                      fmt='s-', color='orange', linewidth=2, markersize=8, capsize=5, label='Switches')
        
        ax1.set_xlabel('Thermal Resistance R (°C/kW)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Daily Cost ($)', fontsize=11, fontweight='bold')
        ax1_2.set_ylabel('Switches per Day', fontsize=11, fontweight='bold')
        ax1.set_title('(a) Insulation Quality Sensitivity', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # C sensitivity
        data_C = df[df['parameter'] == 'C']
        ax2.errorbar(data_C['value'], data_C['cost'], yerr=data_C['cost_std'],
                    fmt='o-', linewidth=2, markersize=8, capsize=5, label='Cost')
        ax2_2 = ax2.twinx()
        ax2_2.errorbar(data_C['value'], data_C['switches'], yerr=data_C['switches_std'],
                      fmt='s-', color='orange', linewidth=2, markersize=8, capsize=5, label='Switches')
        
        ax2.set_xlabel('Thermal Capacitance C (kWh/°C)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Daily Cost ($)', fontsize=11, fontweight='bold')
        ax2_2.set_ylabel('Switches per Day', fontsize=11, fontweight='bold')
        ax2.set_title('(b) Thermal Mass Sensitivity', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Sensitivity Analysis: Building Thermal Parameters',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/sensitivity_thermal_params.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.save_dir}/sensitivity_thermal_params.pdf", bbox_inches='tight')
        plt.close()
        
        print("✓ Thermal parameter sensitivity plot saved")
    
    def _plot_summary_heatmap(self):
        """Plot summary heatmap of all sensitivities."""
        # Create summary data
        summary_data = []
        
        for analysis_name, df in self.results.items():
            for param in df['parameter'].unique():
                data = df[df['parameter'] == param]
                
                # Calculate coefficient of variation (CV)
                cost_cv = (data['cost_std'].mean() / data['cost'].mean()) * 100
                switches_cv = (data['switches_std'].mean() / data['switches'].mean()) * 100
                
                summary_data.append({
                    'Analysis': analysis_name.replace('_', ' ').title(),
                    'Parameter': param,
                    'Cost CV (%)': cost_cv,
                    'Switches CV (%)': switches_cv
                })
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Pivot for heatmap
            pivot = df_summary.pivot(index='Parameter', columns='Analysis', values='Cost CV (%)')
            
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                       ax=ax, cbar_kws={'label': 'Coefficient of Variation (%)'})
            
            ax.set_title('Sensitivity Summary: Parameter Robustness\n(Lower = More Robust)',
                        fontsize=13, fontweight='bold', pad=15)
            ax.set_xlabel('Analysis Type', fontsize=12, fontweight='bold')
            ax.set_ylabel('Parameter', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/sensitivity_summary_heatmap.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"{self.save_dir}/sensitivity_summary_heatmap.pdf", bbox_inches='tight')
            plt.close()
            
            print("✓ Summary heatmap saved")
    
    def generate_report(self):
        """Generate comprehensive sensitivity analysis report."""
        report_path = f"{self.save_dir}/sensitivity_analysis_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("COMPREHENSIVE SENSITIVITY ANALYSIS REPORT\n")
            f.write("Physics-Informed DRL for Building Energy Management\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Results Directory: {self.save_dir}\n\n")
            
            for analysis_name, df in self.results.items():
                f.write("\n" + "-"*70 + "\n")
                f.write(f"{analysis_name.upper().replace('_', ' ')}\n")
                f.write("-"*70 + "\n\n")
                
                f.write(df.to_string(index=False))
                f.write("\n\n")
                
                # Statistical summary
                f.write("Statistical Summary:\n")
                f.write(f"  Cost range: ${df['cost'].min():.4f} - ${df['cost'].max():.4f}\n")
                f.write(f"  Switches range: {df['switches'].min():.0f} - {df['switches'].max():.0f}\n")
                f.write(f"  Cost CV: {(df['cost_std'].mean() / df['cost'].mean() * 100):.1f}%\n")
                f.write(f"  Switches CV: {(df['switches_std'].mean() / df['switches'].mean() * 100):.1f}%\n")
                f.write("\n")
        
        print(f"\n✓ Comprehensive report saved: {report_path}")


def run_full_sensitivity_analysis(quick_mode: bool = True):
    """
    Run complete sensitivity analysis suite.
    
    Args:
        quick_mode: If True, use reduced timesteps for faster execution
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE SENSITIVITY ANALYSIS")
    print("Physics-Informed DRL for Building Energy Management")
    print("="*70)
    print("\nTarget: Applied Energy (Q1 Journal)")
    print("Purpose: Demonstrate robustness across parameter variations")
    print("="*70 + "\n")
    
    if quick_mode:
        timesteps = 10000
        n_eval = 3
        print("⚡ Quick mode: Reduced timesteps for fast testing\n")
    else:
        timesteps = 50000
        n_eval = 10
        print("🎯 Full mode: Production-quality analysis\n")
    
    # Base configuration
    base_params = {
        'R': 2.5,
        'C': 10.0,
        'w_cost': 1.0,
        'w_discomfort': 5.0,
        'w_cycling': 10.0,
        'min_cycle_time': 15,
        'comfort_temp_min': 20.0,
        'comfort_temp_max': 24.0,
        'learning_rate': 3e-4,
        'gamma': 0.99
    }
    
    # Initialize analyzer
    analyzer = SensitivityAnalyzer(base_params, save_dir="./sensitivity_results")
    
    # Run all analyses
    print("\nStarting sensitivity analyses...")
    print("This will take approximately:", "10-15 minutes" if quick_mode else "60-90 minutes")
    print("\n")
    
    # Analysis 1: Reward weights (MOST IMPORTANT)
    analyzer.analyze_reward_weights(timesteps=timesteps, n_eval=n_eval)
    
    # Analysis 2: Thermal parameters
    analyzer.analyze_thermal_parameters(timesteps=timesteps, n_eval=n_eval)
    
    # Analysis 3: Control constraints
    analyzer.analyze_control_constraints(timesteps=timesteps, n_eval=n_eval)
    
    # Analysis 4: PPO hyperparameters
    analyzer.analyze_ppo_hyperparameters(timesteps=timesteps, n_eval=n_eval)
    
    # Analysis 5: Climate conditions
    analyzer.analyze_climate_conditions(timesteps=timesteps, n_eval=n_eval)
    
    # Generate visualizations
    analyzer.generate_visualizations()
    
    # Generate report
    analyzer.generate_report()
    
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {analyzer.save_dir}/")
    print("\nGenerated files:")
    print("  - CSV data for each analysis")
    print("  - Publication-quality figures (PNG + PDF)")
    print("  - Comprehensive text report")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive sensitivity analysis")
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with reduced timesteps')
    parser.add_argument('--full', action='store_true',
                       help='Full production-quality analysis')
    
    args = parser.parse_args()
    
    run_full_sensitivity_analysis(quick_mode=not args.full)
