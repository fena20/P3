# Usage Examples: Physics-Informed DRL Building Energy Management

This document provides practical usage examples for the PI-DRL framework.

## Table of Contents
1. [Basic Environment Usage](#basic-environment-usage)
2. [Training a Custom Agent](#training-a-custom-agent)
3. [Evaluating Performance](#evaluating-performance)
4. [Generating Visualizations](#generating-visualizations)
5. [Advanced Customization](#advanced-customization)

---

## Basic Environment Usage

### Example 1: Running a Simple Episode

```python
from environment import SmartHomeEnv

# Create environment
env = SmartHomeEnv()
obs, info = env.reset()

print("Initial State:", obs)
print("State Components:")
print(f"  Indoor Temp: {obs[0]:.1f}°C")
print(f"  Outdoor Temp: {obs[1]:.1f}°C")
print(f"  Solar Rad: {obs[2]:.1f} W/m²")
print(f"  Price: ${obs[3]:.3f}/kWh")
print(f"  Last Action: {obs[4]}")
print(f"  Time of Day: {obs[5]:.2f}h")

# Run 24-hour episode
for minute in range(1440):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    
    if minute % 60 == 0:  # Print every hour
        print(f"Hour {minute//60}: T_in={info['indoor_temp']:.1f}°C, "
              f"Switches={info['episode_switches']}")
    
    if terminated or truncated:
        break

print(f"\nEpisode Summary:")
print(f"  Total Cost: ${info['episode_cost']:.4f}")
print(f"  Total Discomfort: {info['episode_discomfort']:.2f}")
print(f"  Total Switches: {info['episode_switches']}")
```

### Example 2: Testing the Cycling Penalty

```python
from environment import SmartHomeEnv
import numpy as np

# Create environment with strict cycling penalty
env = SmartHomeEnv()
env.w_cycling = 20.0  # Increase penalty weight

obs, info = env.reset()
rewards_with_cycling = []

# Try to switch frequently
for i in range(60):
    action = i % 2  # Alternate ON/OFF every minute
    obs, reward, terminated, truncated, info = env.step(action)
    rewards_with_cycling.append(reward)
    print(f"Min {i}: Action={action}, Reward={reward:.2f}, "
          f"Time Since Switch={info['time_since_last_switch']}")

print(f"\nMean Reward (frequent switching): {np.mean(rewards_with_cycling):.2f}")

# Now test with no cycling penalty
env.reset()
env.w_cycling = 0.0  # Disable penalty
rewards_no_penalty = []

for i in range(60):
    action = i % 2
    obs, reward, terminated, truncated, info = env.step(action)
    rewards_no_penalty.append(reward)

print(f"Mean Reward (no penalty): {np.mean(rewards_no_penalty):.2f}")
print(f"\nPenalty Impact: {np.mean(rewards_with_cycling) - np.mean(rewards_no_penalty):.2f}")
```

---

## Training a Custom Agent

### Example 3: Quick Training Run

```python
from train_ppo import train_ppo_agent

# Train with minimal timesteps for testing
model, metrics, env = train_ppo_agent(
    total_timesteps=10000,  # Quick test
    n_envs=2,               # Fewer parallel envs
    save_dir="./models_test",
    log_dir="./logs_test"
)

print("Training Complete!")
print(f"Episodes: {len(metrics['episode_rewards'])}")
print(f"Final Mean Reward: {metrics['episode_rewards'][-10:]}")
```

### Example 4: Full Training with Custom Hyperparameters

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment import SmartHomeEnv, load_ampds2_mock_data
from stable_baselines3.common.monitor import Monitor

# Load data
data = load_ampds2_mock_data(num_samples=200000)

# Create custom environment with different parameters
def make_custom_env():
    env = SmartHomeEnv(
        data=data,
        episode_length=2880,  # 48 hours
        comfort_temp_range=(19.0, 25.0),  # Wider comfort range
        min_cycle_time=20  # Stricter cycling (20 min)
    )
    # Adjust reward weights
    env.w_cost = 2.0      # More cost-sensitive
    env.w_discomfort = 3.0
    env.w_cycling = 15.0  # Very strict on cycling
    return Monitor(env)

# Vectorize
env = DummyVecEnv([make_custom_env for _ in range(4)])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Custom PPO configuration
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=5e-4,      # Faster learning
    n_steps=4096,            # More steps per update
    batch_size=128,          # Larger batches
    n_epochs=15,             # More epochs
    gamma=0.995,             # Longer-term rewards
    ent_coef=0.02,           # More exploration
    verbose=1
)

# Train
model.learn(total_timesteps=500000)
model.save("./models/custom_ppo")
env.save("./models/custom_vec_normalize.pkl")
```

---

## Evaluating Performance

### Example 5: Comparing Multiple Models

```python
from stable_baselines3 import PPO
from environment import SmartHomeEnv, load_ampds2_mock_data
import numpy as np

# Load data
data = load_ampds2_mock_data(num_samples=100000)

# Load multiple models
models = {
    'PPO_v1': PPO.load('./models/ppo_v1'),
    'PPO_v2': PPO.load('./models/ppo_v2'),
    'PPO_v3': PPO.load('./models/ppo_v3')
}

# Evaluate each
results = {}
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    env = SmartHomeEnv(data=data, episode_length=1440)
    
    costs = []
    switches = []
    discomforts = []
    
    for episode in range(10):
        obs, _ = env.reset()
        episode_cost = 0
        
        for _ in range(1440):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        
        costs.append(info['episode_cost'])
        switches.append(info['episode_switches'])
        discomforts.append(info['episode_discomfort'])
    
    results[name] = {
        'cost': (np.mean(costs), np.std(costs)),
        'switches': (np.mean(switches), np.std(switches)),
        'discomfort': (np.mean(discomforts), np.std(discomforts))
    }

# Print comparison table
print("\n" + "="*70)
print("MODEL COMPARISON RESULTS")
print("="*70)
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  Cost: ${metrics['cost'][0]:.4f} ± ${metrics['cost'][1]:.4f}")
    print(f"  Switches: {metrics['switches'][0]:.1f} ± {metrics['switches'][1]:.1f}")
    print(f"  Discomfort: {metrics['discomfort'][0]:.2f} ± {metrics['discomfort'][1]:.2f}")
```

### Example 6: Statistical Significance Testing

```python
from scipy.stats import ttest_rel
import numpy as np

# Run baseline and PI-DRL multiple times
baseline_costs = []
pidrl_costs = []

for seed in range(30):
    # Baseline
    baseline_env = BaselineThermostatEnv()
    baseline_env.reset(seed=seed)
    # ... run episode, collect cost
    
    # PI-DRL
    pidrl_env = SmartHomeEnv()
    pidrl_env.reset(seed=seed)
    # ... run episode, collect cost

# Paired t-test
t_stat, p_value = ttest_rel(baseline_costs, pidrl_costs)

print(f"Baseline: ${np.mean(baseline_costs):.4f} ± ${np.std(baseline_costs):.4f}")
print(f"PI-DRL: ${np.mean(pidrl_costs):.4f} ± ${np.std(pidrl_costs):.4f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.6f}")

if p_value < 0.001:
    print("✓ Statistically significant improvement (p < 0.001)")
```

---

## Generating Visualizations

### Example 7: Custom Figure 1 with Different Zoom

```python
from visualizer import ResultVisualizer
import numpy as np

# Assume you have baseline_data and pidrl_data from evaluation
viz = ResultVisualizer(save_dir="./custom_figures")

# Generate heartbeat for different time windows
time_windows = [
    (0, 120, "morning"),     # First 2 hours
    (360, 480, "afternoon"),  # 6-8 AM
    (1020, 1140, "evening")   # 5-7 PM
]

for start, end, label in time_windows:
    viz.figure1_system_heartbeat(
        baseline_actions=baseline_data['actions'],
        baseline_temps=baseline_data['temps'],
        pidrl_actions=pidrl_data['actions'],
        pidrl_temps=pidrl_data['temps'],
        zoom_window=(start, end)
    )
    # Rename the saved figure
    import os
    os.rename(
        "./custom_figures/fig1_system_heartbeat.png",
        f"./custom_figures/fig1_heartbeat_{label}.png"
    )
```

### Example 8: Generating All Figures for Multiple Scenarios

```python
from visualizer import ResultVisualizer

scenarios = [
    ('winter', winter_baseline, winter_pidrl),
    ('summer', summer_baseline, summer_pidrl),
    ('spring', spring_baseline, spring_pidrl)
]

for season, baseline, pidrl in scenarios:
    viz = ResultVisualizer(save_dir=f"./figures_{season}")
    
    # Figure 1
    viz.figure1_system_heartbeat(
        baseline['actions'], baseline['temps'],
        pidrl['actions'], pidrl['temps']
    )
    
    # Figure 3
    viz.figure3_radar_chart(
        baseline_metrics={'Energy Cost': 100, ...},
        pidrl_metrics={'Energy Cost': 78, ...}
    )
    
    # Figure 4
    viz.figure4_energy_carpet_plot(
        baseline['power'], pidrl['power'], num_days=30
    )
    
    print(f"✓ Generated figures for {season}")
```

---

## Advanced Customization

### Example 9: Custom Thermal Model

```python
from environment import SmartHomeEnv

class AdvancedThermalEnv(SmartHomeEnv):
    """Environment with 2nd-order RC model"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add second RC branch
        self.R2 = 5.0
        self.C2 = 20.0
        self.temp_wall = 22.0
    
    def step(self, action):
        # Before physics update
        idx = self.episode_start_idx + self.current_step
        row = self.data.iloc[idx]
        outdoor_temp = row['Outdoor_Temp']
        
        # 2nd-order RC model
        q_out_to_wall = (outdoor_temp - self.temp_wall) / self.R2
        q_wall_to_in = (self.temp_wall - self.indoor_temp) / self.R
        q_hvac = action * self.hvac_power * self.hvac_cop
        q_solar = row['Solar_Rad'] * self.solar_gain_factor
        
        # Update wall temperature
        self.temp_wall += self.dt * (q_out_to_wall - q_wall_to_in) / self.C2
        
        # Update indoor temperature
        self.indoor_temp += self.dt * (q_wall_to_in + q_hvac + q_solar) / self.C
        
        # Continue with original reward calculation
        # ... (rest of step() method)

# Use the advanced model
env = AdvancedThermalEnv()
```

### Example 10: Custom Reward Function for Peak Demand Reduction

```python
from environment import SmartHomeEnv

class PeakShavingEnv(SmartHomeEnv):
    """Environment with additional peak demand penalty"""
    
    def __init__(self, peak_threshold=3.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.peak_threshold = peak_threshold  # kW
        self.w_peak = 5.0  # Peak penalty weight
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Add peak demand penalty
        power = action * self.hvac_power
        if power > self.peak_threshold:
            peak_penalty = (power - self.peak_threshold) ** 2
            reward -= self.w_peak * peak_penalty
            info['peak_penalty'] = peak_penalty
        
        return obs, reward, terminated, truncated, info

# Train with peak shaving objective
env = PeakShavingEnv(peak_threshold=2.5)
```

### Example 11: Multi-Objective Visualization

```python
from visualizer import ResultVisualizer
import numpy as np

# Collect data from multiple weight configurations
configs = [
    {'w_cost': 2, 'w_discomfort': 1, 'w_cycling': 10, 'name': 'Cost-Focused'},
    {'w_cost': 1, 'w_discomfort': 5, 'w_cycling': 10, 'name': 'Comfort-Focused'},
    {'w_cost': 1, 'w_discomfort': 1, 'w_cycling': 20, 'name': 'Longevity-Focused'}
]

results = {}
for config in configs:
    env = SmartHomeEnv()
    env.w_cost = config['w_cost']
    env.w_discomfort = config['w_discomfort']
    env.w_cycling = config['w_cycling']
    
    # Train and evaluate
    model = train_agent(env)
    metrics = evaluate(model, env)
    results[config['name']] = metrics

# Generate comparative radar charts
viz = ResultVisualizer()
for name, metrics in results.items():
    viz.figure3_radar_chart(
        baseline_metrics={'Energy Cost': 100, ...},
        pidrl_metrics=metrics
    )
    # Rename saved figure
    import os
    os.rename(
        "./figures/fig3_radar_chart.png",
        f"./figures/fig3_radar_{name.lower().replace(' ', '_')}.png"
    )
```

### Example 12: Integration with Home Assistant

```python
"""
Example: Deploy trained model in Home Assistant
Requires: hass-pyscript integration
"""

from stable_baselines3 import PPO
import numpy as np

# Load model
model = PPO.load("/config/models/ppo_smarthome_final")

@service
def pidrl_hvac_control():
    """Smart HVAC control using PI-DRL agent"""
    
    # Read sensor states
    indoor_temp = float(state.get('sensor.indoor_temperature'))
    outdoor_temp = float(state.get('sensor.outdoor_temperature'))
    solar_rad = float(state.get('sensor.solar_radiation'))
    
    # Get current electricity price
    price = float(state.get('sensor.electricity_price'))
    
    # Get last action
    last_action = 1 if state.get('climate.hvac') == 'heat' else 0
    
    # Get time of day
    import datetime
    now = datetime.datetime.now()
    time_of_day = now.hour + now.minute / 60.0
    
    # Construct state
    state_vector = np.array([
        indoor_temp, outdoor_temp, solar_rad,
        price, last_action, time_of_day
    ], dtype=np.float32)
    
    # Get action from model
    action, _ = model.predict(state_vector, deterministic=True)
    
    # Send control command
    if action == 1:
        service.call('climate', 'turn_on', entity_id='climate.hvac')
    else:
        service.call('climate', 'turn_off', entity_id='climate.hvac')
    
    log.info(f"PI-DRL Control: T_in={indoor_temp}°C, Action={'ON' if action==1 else 'OFF'}")

# Automation: Call every minute
@time_trigger('cron(* * * * *)')
def run_pidrl_control():
    pidrl_hvac_control()
```

---

## Tips and Best Practices

### Training Tips

1. **Start with quick runs**: Use 10k-30k timesteps to verify code works
2. **Monitor metrics**: Watch for convergence in reward and switches
3. **Use TensorBoard**: `tensorboard --logdir ./logs` for real-time monitoring
4. **Save checkpoints**: Don't lose progress from long training runs
5. **Normalize observations**: Always use `VecNormalize` for stable learning

### Evaluation Tips

1. **Multiple seeds**: Run 20-30 episodes with different seeds
2. **Statistical testing**: Use t-tests for significance
3. **Seasonal variation**: Test on winter, summer, and shoulder seasons
4. **Edge cases**: Test extreme weather conditions
5. **Real-time performance**: Measure inference time (<100ms recommended)

### Visualization Tips

1. **High DPI**: Use 300 DPI for publication figures
2. **Color blindness**: Test figures with colorblind-friendly palettes
3. **Font sizes**: Ensure readability when printed (≥10pt)
4. **Legend placement**: Avoid obscuring data
5. **Captions**: Include descriptive figure captions in manuscript

---

## Troubleshooting

### Common Issues

**Issue 1: Training not converging**
```python
# Solution: Adjust learning rate and entropy coefficient
model = PPO(..., learning_rate=1e-4, ent_coef=0.05)
```

**Issue 2: Too many switches despite penalty**
```python
# Solution: Increase cycling penalty weight
env.w_cycling = 20.0  # or higher
```

**Issue 3: Poor temperature control**
```python
# Solution: Increase discomfort weight
env.w_discomfort = 10.0
```

**Issue 4: Figures not generating**
```python
# Solution: Check matplotlib backend
import matplotlib
matplotlib.use('Agg')  # For headless servers
```

---

## Additional Resources

- **Documentation**: See `IMPLEMENTATION_GUIDE.md` for detailed technical docs
- **Examples**: See `test_integration.py` for working examples
- **Training**: See `train_ppo.py` for full training pipeline
- **Issues**: Open GitHub issue for bugs or questions

---

**Last Updated**: December 2, 2025
