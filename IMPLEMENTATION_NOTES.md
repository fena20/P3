# Implementation Notes: Physics-Informed Deep Reinforcement Learning Framework

## Overview

This document provides detailed explanations of key implementation aspects, particularly focusing on the novel contributions for publication in Applied Energy (Q1 Journal).

## 1. Physics-Informed Thermal Model

### RC Circuit Model Implementation

The environment implements a first-order RC thermal model:

$$T_{in}^{t+1} = T_{in}^t + \Delta t \times \left[\frac{T_{out} - T_{in}}{R} + \frac{Q_{HVAC} + Q_{Solar}}{C}\right]$$

**Key Parameters:**
- `R = 0.05 K/kW`: Thermal resistance (how easily heat flows through building envelope)
- `C = 0.5 kWh/K`: Thermal capacitance (building's ability to store thermal energy)
- `dt = 1/60 hours`: Time step (1 minute resolution, matching AMPds2)
- `Q_hvac_max = 3.0 kW`: Maximum HVAC power output

**Physical Interpretation:**
- The first term `(T_out - T_in)/R` represents heat transfer through the building envelope
- The second term `(Q_HVAC + Q_Solar)/C` represents heat addition from HVAC and solar gains
- The model captures thermal inertia, preventing unrealistic temperature jumps

## 2. Cycling Penalty: Addressing Short-Cycling

### Problem Statement

**Short-cycling** is a critical issue in heat pump systems where the compressor turns on and off too frequently (e.g., every few minutes). This causes:

1. **Mechanical Wear**: Frequent start-stop cycles stress compressor components
2. **Reduced Efficiency**: Startup transients consume extra energy
3. **Equipment Degradation**: Premature failure of electrical and mechanical components
4. **Increased Energy Costs**: Lower overall system efficiency

### Implementation

The cycling penalty enforces a **minimum cycle time** (default: 15 minutes) between state changes:

```python
def _calculate_cycling_penalty(self, action):
    if action != self.last_action:  # State change detected
        time_since_switch = self.current_step - self.last_action_time
        
        if time_since_switch < self.min_cycle_time:
            # Penalty increases with violation severity
            violation_ratio = (self.min_cycle_time - time_since_switch) / self.min_cycle_time
            penalty = violation_ratio * 10.0
            return penalty
```

**Key Features:**
- **Adaptive Penalty**: Penalty scales with violation severity (immediate switch = maximum penalty)
- **Hardware Protection**: 15-minute minimum aligns with manufacturer recommendations
- **Learning Signal**: Agent learns to maintain stable operation periods

### Reward Function Integration

The cycling penalty is integrated into the multi-objective reward:

$$R = -(w_1 \cdot Cost + w_2 \cdot Discomfort + w_3 \cdot Cycling\_Penalty)$$

**Weight Tuning:**
- `w1 = 1.0`: Energy cost weight
- `w2 = 10.0`: Discomfort weight (higher to prioritize comfort)
- `w3 = 5.0`: Cycling penalty weight (significant to prevent short-cycling)

## 3. State Space Design

The 6D state space captures all relevant information for optimal control:

1. **Indoor_Temp** (normalized): Current indoor temperature
2. **Outdoor_Temp** (normalized): External temperature (disturbance)
3. **Solar_Rad** (normalized): Solar radiation (disturbance)
4. **Price** (normalized): Electricity price (for demand response)
5. **Last_Action**: Previous action (for state continuity)
6. **Time_Index**: Normalized time (0-1) for daily/seasonal patterns

**Normalization Benefits:**
- Faster convergence in neural network training
- Better generalization across different operating conditions
- Improved numerical stability

## 4. Baseline Comparison

### Baseline Thermostat Strategy

The baseline implements a simple ON/OFF thermostat:

```python
if current_temp < setpoint - tolerance:
    action = 1  # ON
elif current_temp > setpoint + tolerance:
    action = 0  # OFF
else:
    action = last_action  # Maintain state
```

**Characteristics:**
- No minimum cycle time enforcement
- Frequent switching near setpoint boundaries
- Represents typical residential thermostat behavior
- Demonstrates short-cycling problem

### PI-DRL Agent Advantages

The trained PPO agent learns to:
- **Prevent Short-Cycling**: Maintains minimum cycle times
- **Demand Response**: Shifts load away from peak pricing hours
- **Comfort Optimization**: Balances energy cost and comfort
- **Predictive Control**: Uses time information for anticipatory actions

## 5. Visualization Strategy

### Figure 1: System Heartbeat

**Purpose**: Demonstrate short-cycling prevention at micro-scale

**Key Elements:**
- Dual-axis plot (compressor state + temperature)
- 2-hour zoom-in window
- Direct comparison: Baseline (frequent switching) vs. PI-DRL (stable runs)

**Insight**: Visual proof that PI-DRL maintains longer ON/OFF cycles

### Figure 2: Control Policy Heatmap

**Purpose**: Explainability and demand response visualization

**Key Elements:**
- 2D heatmap: Hour × Outdoor Temperature
- Color represents probability of Action=ON
- Peak pricing hours highlighted (17:00-20:00)

**Insight**: Agent learns to reduce ON probability during peak hours even at high temperatures (demand response)

### Figure 3: Multi-Objective Radar Chart

**Purpose**: Comprehensive performance comparison

**Metrics:**
- Energy Cost (economic)
- Comfort Violation (thermal)
- Equipment Cycles (reliability)
- Peak Load (grid impact)
- Carbon Emissions (environmental)

**Insight**: PI-DRL achieves improvements across all objectives simultaneously

### Figure 4: Energy Carpet Plot

**Purpose**: Load shifting visualization

**Key Elements:**
- 2D heatmap: Day of Year × Hour of Day
- Color represents HVAC power consumption
- Side-by-side comparison: Baseline vs. PI-DRL

**Insight**: "Red zones" (high consumption) shift away from peak pricing hours in optimized version

## 6. Training Configuration

### PPO Hyperparameters

```python
learning_rate = 3e-4      # Standard learning rate
n_steps = 2048           # Steps per update
batch_size = 64          # Mini-batch size
n_epochs = 10            # Optimization epochs per update
gamma = 0.99             # Discount factor
gae_lambda = 0.95        # GAE parameter
clip_range = 0.2         # PPO clip range
ent_coef = 0.01          # Entropy coefficient (exploration)
vf_coef = 0.5            # Value function coefficient
```

### Training Strategy

1. **Initial Training**: 50,000 timesteps (adjustable)
2. **Checkpointing**: Every 10,000 steps
3. **Best Model Saving**: Based on mean episode reward
4. **TensorBoard Logging**: For training monitoring

## 7. Data Handling

### Synthetic Data Generation

Since full AMPds2 dataset may not be available, the code generates realistic synthetic data:

- **Outdoor Temperature**: Sinusoidal with daily and seasonal patterns
- **Solar Radiation**: Zero at night, peak at noon
- **Electricity Price**: Time-of-Use (TOU) pricing with peak hours
- **Load Patterns**: Realistic spikes and variations

**Advantages:**
- Immediate testing without external data
- Reproducible results
- Easy parameter adjustment

**For Real Data**: Simply provide CSV file path to `load_ampds2_data()`

## 8. Publication Quality Standards

### Figure Formatting

- **Font**: Times New Roman, size 12
- **Style**: Seaborn "whitegrid" with paper context
- **Resolution**: 300 DPI minimum
- **Format**: PDF (vector graphics for scalability)
- **Layout**: Tight bounding boxes, proper spacing

### Code Organization

- **Modular Design**: Separate modules for data, environment, training, visualization
- **Documentation**: Comprehensive docstrings and comments
- **Reproducibility**: Fixed random seeds, deterministic options
- **Extensibility**: Easy to modify parameters and add features

## 9. Key Contributions for Publication

1. **Physics-Informed RL**: Integration of RC thermal model into RL framework
2. **Cycling Penalty**: Novel reward shaping to prevent short-cycling
3. **Multi-Objective Optimization**: Simultaneous optimization of cost, comfort, and reliability
4. **Demand Response**: Learned policy adapts to time-varying electricity prices
5. **Comprehensive Evaluation**: Four publication-quality visualizations

## 10. Future Extensions

Potential enhancements for future work:

- **Multi-zone Control**: Extend to multiple rooms/zones
- **Uncertainty Quantification**: Robust control under uncertainty
- **Transfer Learning**: Pre-trained models for different building types
- **Real-time Adaptation**: Online learning for changing conditions
- **Hardware-in-the-Loop**: Validation on physical testbed

## References

- AMPds2 Dataset: Makonin et al., "AMPds2: The Almanac of Minutely Power dataset Series 2"
- PPO Algorithm: Schulman et al., "Proximal Policy Optimization Algorithms"
- RC Thermal Models: Standard building energy modeling approach
- Applied Energy: Target journal for publication
