# Physics-Informed Deep Reinforcement Learning for Residential Building Energy Management

## Overview

This repository implements a **Physics-Informed Deep Reinforcement Learning (PI-DRL)** framework for optimal HVAC control in residential buildings using the AMPds2 dataset. The implementation is designed for publication in **Applied Energy** (Q1 Journal).

## Key Features

### 1. Physics-Informed Environment (`SmartHomeEnv`)
- **State Space**: 6D continuous (Indoor Temp, Outdoor Temp, Solar Radiation, Price, Last Action, Time Index)
- **Action Space**: Discrete(2) - [OFF, ON] for Heat Pump control
- **Physics Model**: First-order RC thermal model:
  $$T_{in}^{t+1} = T_{in}^t + \Delta t \times \left[\frac{T_{out} - T_{in}}{R} + \frac{Q_{HVAC} + Q_{Solar}}{C}\right]$$
- **Reward Function**: Multi-objective optimization with cycling penalty:
  $$R = -(w_1 \cdot Cost + w_2 \cdot Discomfort + w_3 \cdot Cycling\_Penalty)$$

### 2. Cycling Penalty (Short-Cycling Prevention)
The implementation includes a critical **cycling penalty** that prevents switching more than once every 15 minutes. This addresses the "short-cycling" problem in heat pumps, which causes:
- Increased wear on compressor and electrical components
- Reduced efficiency due to startup transients
- Higher energy consumption
- Potential equipment failure

### 3. PPO Agent
- Proximal Policy Optimization (PPO) from `stable-baselines3`
- Automatic model checkpointing and best model saving
- TensorBoard logging for training monitoring

### 4. Publication-Quality Visualizations
Four journal-standard figures:
1. **System Heartbeat**: Micro-dynamics showing short-cycling prevention
2. **Control Policy Heatmap**: Explainability visualization of learned demand response
3. **Multi-Objective Radar Chart**: Performance comparison across metrics
4. **Energy Carpet Plot**: Load shifting visualization

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
python main.py
```

This will:
1. Generate synthetic AMPds2-like data
2. Train a PPO agent on the SmartHomeEnv
3. Run baseline comparison
4. Generate all publication-quality figures

### Training Only

```bash
python train_ppo.py
```

### Custom Training

```python
from train_ppo import train_ppo

model, env = train_ppo(
    total_timesteps=100000,
    learning_rate=3e-4,
    save_path="./models/my_model"
)
```

### Generate Visualizations

```python
from visualizer import ResultVisualizer
from stable_baselines3 import PPO

# Load trained agent
agent = PPO.load("./models/ppo_smarthome/best_model")

# Create visualizer
viz = ResultVisualizer()

# Generate individual figures
viz.figure1_system_heartbeat(piddrl_data, baseline_data, save_path="fig1.pdf")
viz.figure2_control_policy_heatmap(agent, env, save_path="fig2.pdf")
viz.figure3_multi_objective_radar(baseline_metrics, piddrl_metrics, save_path="fig3.pdf")
viz.figure4_energy_carpet_plot(baseline_power, piddrl_power, save_path="fig4.pdf")
```

## Project Structure

```
.
├── data_loader.py          # AMPds2 data loading and synthetic data generation
├── smarthome_env.py        # Physics-informed Gymnasium environment
├── train_ppo.py           # PPO training script with callbacks
├── visualizer.py          # Publication-quality visualization module
├── main.py                # Main execution script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Environment Parameters

The `SmartHomeEnv` can be customized with the following parameters:

- `T_setpoint`: Comfort setpoint temperature (°C), default: 22.0
- `T_tolerance`: Comfort tolerance (°C), default: 1.5
- `R`: Thermal resistance (K/kW), default: 0.05
- `C`: Thermal capacitance (kWh/K), default: 0.5
- `Q_hvac_max`: Maximum HVAC power (kW), default: 3.0
- `min_cycle_time`: Minimum cycle time in minutes, default: 15
- `w1`, `w2`, `w3`: Reward weights for cost, discomfort, and cycling penalty

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper_2024,
  title={Physics-Informed Deep Reinforcement Learning for Residential Building Energy Management},
  author={Your Name},
  journal={Applied Energy},
  year={2024}
}
```

## License

This project is licensed under the MIT License.

## Contact

For questions or issues, please open an issue on GitHub.
