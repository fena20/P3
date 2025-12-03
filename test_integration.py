"""
Quick Integration Test for the Complete Pipeline
=================================================
Tests all components without full training.
"""

import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from environment import SmartHomeEnv, BaselineThermostatEnv, load_ampds2_mock_data
from visualizer import ResultVisualizer

print("="*70)
print("INTEGRATION TEST: Physics-Informed DRL Building Energy Management")
print("="*70 + "\n")

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Test 1: Data Loading
print("Test 1: Loading AMPds2 Data...")
data = load_ampds2_mock_data(num_samples=10000)
assert len(data) == 10000, "Data loading failed"
assert 'Outdoor_Temp' in data.columns, "Missing required column"
print("✓ Data loading successful\n")

# Test 2: Environment Creation
print("Test 2: Creating Environments...")
env = SmartHomeEnv(data=data, episode_length=100)
baseline_env = BaselineThermostatEnv(data=data, episode_length=100)
print("✓ Environments created successfully\n")

# Test 3: Run Episodes
print("Test 3: Running Test Episodes...")

# Baseline episode
obs, info = baseline_env.reset()
baseline_actions = []
baseline_temps = []
for i in range(100):
    action = baseline_env.get_baseline_action()
    obs, reward, terminated, truncated, info = baseline_env.step(action)
    baseline_actions.append(action)
    baseline_temps.append(info['indoor_temp'])
    if terminated or truncated:
        break

baseline_switches = info['episode_switches']
print(f"✓ Baseline episode: {baseline_switches} switches")

# Random policy episode (simulating PI-DRL)
obs, info = env.reset()
pidrl_actions = []
pidrl_temps = []
for i in range(100):
    # Use less frequent random switching to simulate trained agent
    if i == 0 or (i > 0 and pidrl_actions[-1] == 1 and i % 20 < 15):
        action = 1
    elif i > 0 and pidrl_actions[-1] == 0 and i % 20 >= 15:
        action = 1
    else:
        action = 0
    
    obs, reward, terminated, truncated, info = env.step(action)
    pidrl_actions.append(action)
    pidrl_temps.append(info['indoor_temp'])
    if terminated or truncated:
        break

pidrl_switches = info['episode_switches']
print(f"✓ PI-DRL episode: {pidrl_switches} switches\n")

# Test 4: Visualizations
print("Test 4: Generating Visualizations...")
viz = ResultVisualizer(save_dir="./figures")

# Figure 1
viz.figure1_system_heartbeat(
    baseline_actions=np.array(baseline_actions),
    baseline_temps=np.array(baseline_temps),
    pidrl_actions=np.array(pidrl_actions),
    pidrl_temps=np.array(pidrl_temps),
    zoom_window=(0, min(100, len(baseline_actions)))
)

# Figure 3
viz.figure3_radar_chart(
    baseline_metrics={'Energy Cost': 100, 'Comfort\nViolation': 100,
                     'Equipment\nCycles': 100, 'Peak Load': 100, 'Carbon\nEmissions': 100},
    pidrl_metrics={'Energy Cost': 78, 'Comfort\nViolation': 65,
                  'Equipment\nCycles': 42, 'Peak Load': 85, 'Carbon\nEmissions': 75}
)

# Figure 4
baseline_power = np.array(baseline_actions) * 3.5
pidrl_power = np.array(pidrl_actions) * 3.5
# Extend to sufficient length for 7 days
days = 7
target_len = days * 1440
baseline_power_extended = np.tile(baseline_power, int(np.ceil(target_len / len(baseline_power))))[:target_len]
pidrl_power_extended = np.tile(pidrl_power, int(np.ceil(target_len / len(pidrl_power))))[:target_len]

viz.figure4_energy_carpet_plot(
    baseline_power=baseline_power_extended,
    pidrl_power=pidrl_power_extended,
    num_days=days
)

print("\n" + "="*70)
print("ALL INTEGRATION TESTS PASSED!")
print("="*70)
print("\nGenerated files:")
print("  - ./figures/fig1_system_heartbeat.png (and .pdf)")
print("  - ./figures/fig3_radar_chart.png (and .pdf)")
print("  - ./figures/fig4_energy_carpet.png (and .pdf)")
print("\nNext steps:")
print("  1. Install remaining dependencies: pip install stable-baselines3 torch")
print("  2. Run full demo: python3 main_demo.py")
print("  3. For production: Train for 1M+ timesteps on full AMPds2 dataset")
print("="*70 + "\n")
