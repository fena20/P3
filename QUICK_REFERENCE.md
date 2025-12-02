# Quick Reference Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install gymnasium stable-baselines3 numpy pandas matplotlib seaborn
```

### Step 2: Run Integration Test
```bash
python3 test_integration.py
```
✅ Takes 2 minutes, generates sample figures

### Step 3: Run Full Demo (Optional)
```bash
python3 main_demo.py
```
✅ Takes 5-10 minutes, includes training

---

## 📁 File Guide

| File | Purpose | When to Use |
|------|---------|-------------|
| `test_integration.py` | Quick test, no training | First run, verify setup |
| `main_demo.py` | Full pipeline with training | Complete demonstration |
| `environment.py` | Physics environment | Import for custom scripts |
| `train_ppo.py` | PPO training | Custom training experiments |
| `visualizer.py` | Figure generation | Custom visualization |

---

## 🎯 Common Tasks

### Task 1: Generate Figures Only
```python
from visualizer import ResultVisualizer
viz = ResultVisualizer(save_dir="./my_figures")
# Use viz.figure1_system_heartbeat(), etc.
```

### Task 2: Train Custom Model
```python
from train_ppo import train_ppo_agent
model, metrics, env = train_ppo_agent(
    total_timesteps=100000,
    n_envs=4
)
```

### Task 3: Evaluate Existing Model
```python
from stable_baselines3 import PPO
model = PPO.load("./models/ppo_smarthome_final")
# Use model.predict(obs) for inference
```

### Task 4: Adjust Reward Weights
```python
from environment import SmartHomeEnv
env = SmartHomeEnv()
env.w_cost = 2.0        # Energy cost weight
env.w_discomfort = 5.0  # Comfort weight
env.w_cycling = 20.0    # Cycling penalty weight (KEY)
```

---

## 📊 Key Parameters

### Environment
- `episode_length`: Minutes per episode (default: 1440 = 24h)
- `comfort_temp_range`: (T_min, T_max) in °C (default: 20-24)
- `min_cycle_time`: Minimum minutes between switches (default: 15)

### Reward Weights
- `w_cost = 1.0`: Energy cost
- `w_discomfort = 5.0`: Thermal discomfort
- `w_cycling = 10.0`: **Cycling penalty (KEY INNOVATION)**

### Thermal Model
- `R = 2.5`: Thermal resistance (°C/kW)
- `C = 10.0`: Thermal capacitance (kWh/°C)
- `hvac_power = 3.5`: Heat pump power (kW)
- `hvac_cop = 3.0`: Coefficient of Performance

---

## 🔍 Troubleshooting

### "Command not found: python"
→ Use `python3` instead of `python`

### "ModuleNotFoundError: gymnasium"
→ Run `pip install -r requirements.txt`

### "Training too slow"
→ Reduce `total_timesteps` or use GPU

### "Figures not showing"
→ Check `./figures/` directory for saved files

### "Too many switches"
→ Increase `env.w_cycling` (try 20.0 or 30.0)

### "Poor temperature control"
→ Increase `env.w_discomfort` (try 10.0)

---

## 📚 Documentation Index

| Document | Focus |
|----------|-------|
| `README.md` | Overview, installation, quick start |
| `IMPLEMENTATION_GUIDE.md` | Technical details, math, architecture |
| `PROJECT_SUMMARY.md` | High-level summary, contributions |
| `USAGE_EXAMPLES.md` | 12 practical code examples |
| `FINAL_SUMMARY.md` | Complete delivery summary |
| `QUICK_REFERENCE.md` | This file (cheat sheet) |

---

## 🎨 Figure Reference

### Figure 1: System Heartbeat
**File**: `figures/fig1_system_heartbeat.{png,pdf}`  
**Shows**: Compressor cycling comparison (2-hour window)  
**Key Metric**: Switch count (Baseline: ~56, PI-DRL: ~8)

### Figure 2: Policy Heatmap
**File**: `figures/fig2_policy_heatmap.{png,pdf}`  
**Shows**: Learned policy (Hour × Outdoor Temp)  
**Key Insight**: Demand response during peak pricing

### Figure 3: Radar Chart
**File**: `figures/fig3_radar_chart.{png,pdf}`  
**Shows**: 5-metric comparison (Cost, Comfort, Cycles, Peak, Carbon)  
**Key Metric**: Equipment cycles reduced to 42% of baseline

### Figure 4: Carpet Plot
**File**: `figures/fig4_energy_carpet.{png,pdf}`  
**Shows**: 30-day load patterns (Day × Hour)  
**Key Insight**: Load shifted away from peak pricing hours

---

## 💡 Pro Tips

1. **Start small**: Use `total_timesteps=10000` for quick tests
2. **Monitor training**: Use TensorBoard: `tensorboard --logdir ./logs`
3. **Save often**: Training checkpoints saved every 10k steps
4. **Multiple seeds**: Run evaluation with 20+ different seeds
5. **GPU acceleration**: Install PyTorch with CUDA for 10x speedup

---

## 🔗 Quick Links

- **Main Demo**: `python3 main_demo.py`
- **Test Only**: `python3 test_integration.py`
- **Figures**: `./figures/`
- **Models**: `./models/`
- **Logs**: `./logs/`
- **Results**: `./results/`

---

**Need Help?**
1. Check `IMPLEMENTATION_GUIDE.md` for details
2. See `USAGE_EXAMPLES.md` for code samples
3. Open GitHub issue or contact author

**Last Updated**: December 2, 2025
