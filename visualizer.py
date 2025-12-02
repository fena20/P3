"""
Advanced Visualization Module for Publication-Quality Figures
Journal Standard: Applied Energy (Q1)
Style: Times New Roman, size 12, seaborn-paper style
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')


class ResultVisualizer:
    """
    Publication-quality visualization for PI-DRL results
    Generates 4 key figures for journal submission
    """
    
    def __init__(self, figsize=(8, 6), dpi=300):
        """
        Initialize visualizer with publication settings
        
        Parameters:
        -----------
        figsize : tuple
            Default figure size (width, height) in inches
        dpi : int
            Resolution for publication (300 DPI minimum)
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # Set publication style
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.0)
        
        # Configure matplotlib for Times New Roman
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 11
        plt.rcParams['ytick.labelsize'] = 11
        plt.rcParams['legend.fontsize'] = 11
        plt.rcParams['figure.titlesize'] = 16
        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['grid.linewidth'] = 0.5
        
    def figure1_system_heartbeat(self, piddrl_data, baseline_data, 
                                  start_hour=10, duration_hours=2, save_path=None):
        """
        Fig 1: The "System Heartbeat" (Micro-Dynamics)
        
        Shows prevention of short-cycling with dual-axis plot:
        - Left Y: Compressor State (0/1 binary step plot)
        - Right Y: Indoor Temperature
        - Comparison: Baseline vs PI-DRL Agent
        
        Parameters:
        -----------
        piddrl_data : dict
            PI-DRL episode data with 'actions' and 'states' keys
        baseline_data : dict
            Baseline thermostat data (same structure)
        start_hour : int
            Starting hour for zoom-in (0-23)
        duration_hours : int
            Duration of zoom-in window
        save_path : str, optional
            Path to save figure
        """
        fig, ax1 = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Convert to minutes
        start_min = start_hour * 60
        end_min = start_min + duration_hours * 60
        
        # Extract PI-DRL data
        piddrl_actions = np.array(piddrl_data['actions'][start_min:end_min])
        piddrl_temps = np.array([s['indoor_temp'] for s in piddrl_data['states'][start_min:end_min]])
        piddrl_time = np.arange(len(piddrl_actions))
        
        # Extract baseline data (simulate frequent switching)
        baseline_actions = np.array(baseline_data['actions'][start_min:end_min])
        baseline_temps = np.array([s['indoor_temp'] for s in baseline_data['states'][start_min:end_min]])
        baseline_time = np.arange(len(baseline_actions))
        
        # Left Y-axis: Compressor State (Binary Step Plot)
        color1 = '#2E86AB'  # Blue
        color2 = '#A23B72'  # Magenta
        
        ax1.set_xlabel('Time (minutes)', fontsize=12, fontname='Times New Roman')
        ax1.set_ylabel('Compressor State (ON/OFF)', fontsize=12, fontname='Times New Roman', color=color1)
        
        # Plot baseline (frequent switching - typical thermostat behavior)
        ax1.step(baseline_time, baseline_actions, where='post', 
                label='Baseline Thermostat', color=color2, linewidth=1.5, alpha=0.7, linestyle='--')
        
        # Plot PI-DRL (stable runs)
        ax1.step(piddrl_time, piddrl_actions, where='post', 
                label='PI-DRL Agent', color=color1, linewidth=2.0, alpha=0.9)
        
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim([-0.1, 1.1])
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['OFF', 'ON'])
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        
        # Right Y-axis: Indoor Temperature
        ax2 = ax1.twinx()
        ax2.set_ylabel('Indoor Temperature (°C)', fontsize=12, fontname='Times New Roman', color='#F18F01')
        ax2.plot(baseline_time, baseline_temps, color=color2, linewidth=1.5, 
                alpha=0.7, linestyle='--', label='Baseline Temp')
        ax2.plot(piddrl_time, piddrl_temps, color='#F18F01', linewidth=2.0, 
                alpha=0.9, label='PI-DRL Temp')
        ax2.tick_params(axis='y', labelcolor='#F18F01')
        ax2.grid(False)
        
        # Add setpoint line
        setpoint = 22.0
        ax2.axhline(y=setpoint, color='green', linestyle=':', linewidth=1.5, 
                   alpha=0.7, label=f'Setpoint ({setpoint}°C)')
        ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        plt.title('System Heartbeat: Prevention of Short-Cycling', 
                 fontsize=14, fontname='Times New Roman', fontweight='bold', pad=15)
        plt.tight_layout()
        
        if save_path:
            # Detect format from extension
            fmt = 'pdf' if save_path.endswith('.pdf') else 'png'
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', format=fmt)
            print(f"Figure 1 saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def figure2_control_policy_heatmap(self, agent, env, save_path=None):
        """
        Fig 2: Control Policy Heatmap (Explainability)
        
        Shows learned policy as 2D heatmap:
        - X-axis: Hour of Day (0-23)
        - Y-axis: Outdoor Temperature (-5 to 35°C)
        - Color: Probability of Action=ON
        
        Demonstrates demand response during peak pricing hours.
        
        Parameters:
        -----------
        agent : stable_baselines3.PPO
            Trained PPO agent
        env : SmartHomeEnv
            Environment instance
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Create grid for heatmap
        hours = np.arange(0, 24)
        outdoor_temps = np.linspace(-5, 35, 40)
        
        # Initialize probability matrix
        prob_matrix = np.zeros((len(outdoor_temps), len(hours)))
        
        # Sample policy for each (hour, outdoor_temp) combination
        for i, temp in enumerate(outdoor_temps):
            for j, hour in enumerate(hours):
                # Create observation vector
                # [Indoor_Temp, Outdoor_Temp, Solar_Rad, Price, Last_Action, Time_Index]
                indoor_temp = 22.0  # Assume near setpoint
                solar_rad = max(0, 800 * np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0
                price = 0.15 if 17 <= hour < 20 else 0.10  # Peak pricing
                last_action = 0
                time_index = hour / 24.0
                
                obs = np.array([
                    indoor_temp / 30.0,
                    temp / 40.0,
                    solar_rad / 1000.0,
                    price / 1.0,
                    float(last_action),
                    time_index
                ], dtype=np.float32)
                
                # Get action probability from agent
                # Sample multiple times to estimate probability
                n_samples = 50
                actions_on = 0
                for _ in range(n_samples):
                    action, _ = agent.predict(obs, deterministic=False)
                    # Convert action to scalar if needed
                    if isinstance(action, np.ndarray):
                        action = action.item() if action.size == 1 else action[0]
                    if action == 1:
                        actions_on += 1
                prob_on = actions_on / n_samples
                
                prob_matrix[i, j] = prob_on
        
        # Create heatmap
        im = ax.imshow(prob_matrix, aspect='auto', origin='lower', 
                      cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='bilinear')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(0, 24, 2))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)])
        ax.set_xlabel('Hour of Day', fontsize=12, fontname='Times New Roman')
        
        temp_indices = np.linspace(0, len(outdoor_temps)-1, 9).astype(int)
        ax.set_yticks(temp_indices)
        ax.set_yticklabels([f'{outdoor_temps[i]:.0f}' for i in temp_indices])
        ax.set_ylabel('Outdoor Temperature (°C)', fontsize=12, fontname='Times New Roman')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Probability of Action=ON', fontsize=12, fontname='Times New Roman', rotation=270, labelpad=20)
        
        # Highlight peak pricing hours
        ax.axvspan(17, 20, alpha=0.2, color='red', label='Peak Pricing Hours')
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        
        plt.title('Control Policy Heatmap: Learned Demand Response Strategy', 
                 fontsize=14, fontname='Times New Roman', fontweight='bold', pad=15)
        plt.tight_layout()
        
        if save_path:
            # Detect format from extension
            fmt = 'pdf' if save_path.endswith('.pdf') else 'png'
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', format=fmt)
            print(f"Figure 2 saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def figure3_multi_objective_radar(self, baseline_metrics, piddrl_metrics, save_path=None):
        """
        Fig 3: Multi-Objective Radar Chart
        
        Compares baseline vs PI-DRL across multiple objectives:
        - Energy Cost
        - Comfort Violation
        - Equipment Cycles
        - Peak Load
        - Carbon Emissions
        
        Parameters:
        -----------
        baseline_metrics : dict
            Dictionary with keys: ['cost', 'comfort', 'cycles', 'peak_load', 'carbon']
        piddrl_metrics : dict
            Same structure as baseline_metrics
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi, subplot_kw=dict(projection='polar'))
        
        # Categories
        categories = ['Energy\nCost', 'Comfort\nViolation', 'Equipment\nCycles', 
                     'Peak\nLoad', 'Carbon\nEmissions']
        N = len(categories)
        
        # Normalize metrics (baseline = 100%)
        baseline_values = np.array([
            baseline_metrics['cost'],
            baseline_metrics['comfort'],
            baseline_metrics['cycles'],
            baseline_metrics['peak_load'],
            baseline_metrics['carbon']
        ])
        
        piddrl_values = np.array([
            piddrl_metrics['cost'],
            piddrl_metrics['comfort'],
            piddrl_metrics['cycles'],
            piddrl_metrics['peak_load'],
            piddrl_metrics['carbon']
        ])
        
        # Normalize to percentage (baseline = 100%)
        baseline_normalized = (baseline_values / baseline_values) * 100
        piddrl_normalized = (piddrl_values / baseline_values) * 100
        
        # Compute angle for each category
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Complete the values arrays
        baseline_normalized = np.concatenate((baseline_normalized, [baseline_normalized[0]]))
        piddrl_normalized = np.concatenate((piddrl_normalized, [piddrl_normalized[0]]))
        
        # Plot
        ax.plot(angles, baseline_normalized, 'o-', linewidth=2, label='Baseline', 
               color='#A23B72', markersize=8)
        ax.fill(angles, baseline_normalized, alpha=0.15, color='#A23B72')
        
        ax.plot(angles, piddrl_normalized, 'o-', linewidth=2, label='Proposed PI-DRL', 
               color='#2E86AB', markersize=8)
        ax.fill(angles, piddrl_normalized, alpha=0.25, color='#2E86AB')
        
        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11, fontname='Times New Roman')
        
        # Set radial limits
        ax.set_ylim(0, 120)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), 
                 frameon=True, fancybox=True, shadow=True, fontsize=11)
        
        plt.title('Multi-Objective Performance Comparison', 
                 fontsize=14, fontname='Times New Roman', fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            # Detect format from extension
            fmt = 'pdf' if save_path.endswith('.pdf') else 'png'
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', format=fmt)
            print(f"Figure 3 saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def figure4_energy_carpet_plot(self, baseline_power, piddrl_power, save_path=None):
        """
        Fig 4: Energy Carpet Plot (Load Shifting Visualization)
        
        Shows HVAC power consumption as 2D heatmap:
        - X-axis: Day of Year
        - Y-axis: Hour of Day
        - Color: HVAC Power Consumption
        
        Visualizes load shifting away from peak pricing hours.
        
        Parameters:
        -----------
        baseline_power : np.ndarray
            2D array of shape (n_days, 24) with baseline power consumption
        piddrl_power : np.ndarray
            2D array of shape (n_days, 24) with PI-DRL power consumption
        save_path : str, optional
            Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=self.dpi)
        
        # Determine color scale
        vmin = min(baseline_power.min(), piddrl_power.min())
        vmax = max(baseline_power.max(), piddrl_power.max())
        
        # Baseline plot
        im1 = ax1.imshow(baseline_power.T, aspect='auto', origin='lower', 
                        cmap='YlOrRd', vmin=vmin, vmax=vmax, interpolation='bilinear')
        ax1.set_xlabel('Day of Year', fontsize=12, fontname='Times New Roman')
        ax1.set_ylabel('Hour of Day', fontsize=12, fontname='Times New Roman')
        ax1.set_title('Baseline Thermostat', fontsize=13, fontname='Times New Roman', fontweight='bold')
        ax1.set_yticks(np.arange(0, 24, 4))
        ax1.set_yticklabels([f'{h:02d}:00' for h in range(0, 24, 4)])
        
        # Highlight peak pricing hours
        for i in range(baseline_power.shape[0]):
            ax1.axhspan(17, 20, alpha=0.15, color='red', zorder=0)
        
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('HVAC Power (kW)', fontsize=11, fontname='Times New Roman', rotation=270, labelpad=15)
        
        # PI-DRL plot
        im2 = ax2.imshow(piddrl_power.T, aspect='auto', origin='lower', 
                        cmap='YlOrRd', vmin=vmin, vmax=vmax, interpolation='bilinear')
        ax2.set_xlabel('Day of Year', fontsize=12, fontname='Times New Roman')
        ax2.set_ylabel('Hour of Day', fontsize=12, fontname='Times New Roman')
        ax2.set_title('Proposed PI-DRL Agent', fontsize=13, fontname='Times New Roman', fontweight='bold')
        ax2.set_yticks(np.arange(0, 24, 4))
        ax2.set_yticklabels([f'{h:02d}:00' for h in range(0, 24, 4)])
        
        # Highlight peak pricing hours
        for i in range(piddrl_power.shape[0]):
            ax2.axhspan(17, 20, alpha=0.15, color='red', zorder=0, label='Peak Pricing')
        
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('HVAC Power (kW)', fontsize=11, fontname='Times New Roman', rotation=270, labelpad=15)
        
        # Add legend for peak pricing
        ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=10)
        
        plt.suptitle('Energy Carpet Plot: Load Shifting Visualization', 
                    fontsize=14, fontname='Times New Roman', fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            # Detect format from extension
            fmt = 'pdf' if save_path.endswith('.pdf') else 'png'
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', format=fmt)
            print(f"Figure 4 saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_all_figures(self, agent, env, piddrl_data, baseline_data, 
                            baseline_metrics, piddrl_metrics, 
                            baseline_power, piddrl_power, save_dir='./figures'):
        """
        Generate all 4 publication-quality figures
        
        Parameters:
        -----------
        agent : stable_baselines3.PPO
            Trained PPO agent
        env : SmartHomeEnv
            Environment instance
        piddrl_data : dict
            PI-DRL episode data
        baseline_data : dict
            Baseline episode data
        baseline_metrics : dict
            Baseline performance metrics
        piddrl_metrics : dict
            PI-DRL performance metrics
        baseline_power : np.ndarray
            Baseline power consumption matrix
        piddrl_power : np.ndarray
            PI-DRL power consumption matrix
        save_dir : str
            Directory to save figures
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("Generating publication-quality figures...")
        
        self.figure1_system_heartbeat(
            piddrl_data, baseline_data, 
            start_hour=10, duration_hours=2,
            save_path=os.path.join(save_dir, 'figure1_system_heartbeat.pdf')
        )
        
        self.figure2_control_policy_heatmap(
            agent, env,
            save_path=os.path.join(save_dir, 'figure2_policy_heatmap.pdf')
        )
        
        self.figure3_multi_objective_radar(
            baseline_metrics, piddrl_metrics,
            save_path=os.path.join(save_dir, 'figure3_radar_chart.pdf')
        )
        
        self.figure4_energy_carpet_plot(
            baseline_power, piddrl_power,
            save_path=os.path.join(save_dir, 'figure4_energy_carpet.pdf')
        )
        
        print(f"\nAll figures saved to {save_dir}/")
