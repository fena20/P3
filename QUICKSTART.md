# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Test the Environment

First, verify the environment works correctly:

```bash
python test_environment.py
```

This will:
- Test environment reset and step functions
- Verify cycling penalty mechanism
- Run a short episode

### 2. Train the PPO Agent

Train a PPO agent on the SmartHomeEnv:

```bash
python train_ppo.py
```

Or use the main script which includes training:

```bash
python main.py
```

**Training Parameters:**
- Default: 50,000 timesteps
- Models saved to `./models/ppo_smarthome/`
- Best model saved as `best_model.zip`
- Checkpoints every 10,000 steps

### 3. Generate Visualizations

The main script automatically generates all 4 publication-quality figures:

```bash
python main.py
```

Figures are saved to `./figures/`:
- `figure1_system_heartbeat.pdf` - Micro-dynamics showing short-cycling prevention
- `figure2_policy_heatmap.pdf` - Control policy explainability
- `figure3_radar_chart.pdf` - Multi-objective performance comparison
- `figure4_energy_carpet.pdf` - Load shifting visualization

## Customization

### Modify Environment Parameters

Edit `smarthome_env.py` or pass parameters when creating the environment:

```python
from smarthome_env import SmartHomeEnv

env = SmartHomeEnv(
    T_setpoint=23.0,      # Change comfort setpoint
    min_cycle_time=20,    # Change minimum cycle time (minutes)
    w1=1.0,               # Cost weight
    w2=10.0,              # Discomfort weight
    w3=5.0                # Cycling penalty weight
)
```

### Modify Training Configuration

Edit `train_ppo.py` or modify the `train_ppo()` function call:

```python
model, env = train_ppo(
    total_timesteps=100000,  # More training steps
    learning_rate=1e-4,      # Lower learning rate
    save_path="./models/my_model"
)
```

### Use Real AMPds2 Data

Replace synthetic data with real AMPds2 dataset:

```python
from data_loader import load_ampds2_data

# Load from CSV file
data = load_ampds2_data(filepath="path/to/ampds2_data.csv")

# Or modify data_loader.py to load from your data source
```

## Understanding the Output

### Training Output

During training, you'll see:
- Episode rewards
- Best model checkpoints
- TensorBoard logs (in `./tensorboard_logs/`)

### Evaluation Metrics

After running `main.py`, you'll see:
- **Baseline Performance**: Cost, discomfort, cycles
- **PI-DRL Performance**: Improved metrics
- **Comparison**: Side-by-side performance

### Figures Explained

1. **Figure 1**: Shows how PI-DRL prevents frequent switching (short-cycling)
2. **Figure 2**: Visualizes when the agent chooses to turn ON/OFF based on time and temperature
3. **Figure 3**: Compares overall performance across multiple objectives
4. **Figure 4**: Shows how energy consumption shifts away from peak hours

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed (`pip install -r requirements.txt`)

2. **CUDA/GPU Issues**: Stable-baselines3 will use CPU if GPU not available (works fine)

3. **Memory Issues**: Reduce `total_timesteps` or `n_steps` in training configuration

4. **Figure Generation Errors**: Ensure matplotlib backend supports PDF export

### Getting Help

- Check `IMPLEMENTATION_NOTES.md` for detailed explanations
- Review code comments in each module
- Test individual components using `test_environment.py`

## Next Steps

1. **Experiment with Parameters**: Try different reward weights, cycle times, etc.
2. **Extend the Environment**: Add more appliances, multi-zone control, etc.
3. **Compare Algorithms**: Try other RL algorithms (SAC, TD3, etc.)
4. **Real Data Integration**: Connect to actual AMPds2 or building data
5. **Hardware Deployment**: Deploy trained model to real building control system

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
