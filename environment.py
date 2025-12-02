"""
Physics-Informed Smart Home Environment for Building Energy Management
===============================================================
This module implements a custom Gymnasium environment for residential HVAC control
using a first-order RC thermal model based on AMPds2 dataset patterns.

Author: Lead Researcher, Cyber-Physical Energy Systems
Target: Applied Energy (Q1 Journal)
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional


def load_ampds2_mock_data(num_samples: int = 50000) -> pd.DataFrame:
    """
    Generate synthetic AMPds2-like data at 1-minute resolution.
    
    The real AMPds2 dataset contains power consumption data from a Canadian home
    over 2 years at 1-minute resolution. This mock function generates realistic
    patterns for testing and demonstration.
    
    Args:
        num_samples: Number of 1-minute samples (default: ~35 days)
    
    Returns:
        DataFrame with columns: WHE (Water Heater Energy), HPE (Heat Pump Energy),
        FRE (Fridge Energy), Outdoor_Temp, Solar_Rad, Price
    """
    np.random.seed(42)
    
    # Time index (1-minute resolution)
    time_idx = pd.date_range('2024-01-01', periods=num_samples, freq='1min')
    
    # Hour of day (0-23) and day of year for seasonal patterns
    hour_of_day = time_idx.hour + time_idx.minute / 60.0
    day_of_year = time_idx.dayofyear
    
    # Outdoor temperature with diurnal and seasonal variation
    # Winter base temp: -5°C, Summer base: 25°C
    seasonal_temp = 10 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    diurnal_variation = 5 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    outdoor_temp = seasonal_temp + diurnal_variation + np.random.normal(0, 2, num_samples)
    
    # Solar radiation (W/m²) - peak at noon, zero at night
    solar_rad = np.maximum(0, 800 * np.sin(np.pi * (hour_of_day - 6) / 12) * 
                           (1 + 0.3 * np.sin(2 * np.pi * (day_of_year - 172) / 365)))
    # Convert hour_of_day to numpy array for proper indexing
    hour_array = np.array(hour_of_day)
    solar_rad = np.array(solar_rad)
    solar_rad[hour_array < 6] = 0
    solar_rad[hour_array > 18] = 0
    solar_rad += np.random.normal(0, 50, num_samples).clip(0)
    
    # Time-of-Use electricity pricing ($/kWh)
    # Peak: 17:00-20:00, Mid-Peak: 11:00-17:00 & 20:00-22:00, Off-Peak: rest
    price = np.ones(num_samples) * 0.082  # Off-peak base
    peak_hours = (hour_array >= 17) & (hour_array < 20)
    mid_peak_hours = ((hour_array >= 11) & (hour_array < 17)) | \
                     ((hour_array >= 20) & (hour_array < 22))
    price[peak_hours] = 0.180  # Peak price
    price[mid_peak_hours] = 0.113  # Mid-peak price
    
    # Heat Pump Energy (initially zero, will be controlled by agent)
    HPE = np.zeros(num_samples)
    
    # Water Heater Energy (W) - cyclic operation
    WHE = 3000 * (np.random.random(num_samples) < 0.05)  # 5% duty cycle
    
    # Fridge Energy (W) - constant with cycling
    FRE = 150 + 50 * np.sin(2 * np.pi * np.arange(num_samples) / 120) + \
          np.random.normal(0, 20, num_samples)
    
    df = pd.DataFrame({
        'timestamp': time_idx,
        'WHE': WHE,
        'HPE': HPE,
        'FRE': FRE,
        'Outdoor_Temp': outdoor_temp,
        'Solar_Rad': solar_rad,
        'Price': price
    })
    
    return df


class SmartHomeEnv(gym.Env):
    """
    Physics-Informed Smart Home Environment with RC Thermal Model.
    
    This environment simulates a residential building with HVAC control,
    incorporating first-order thermal dynamics and a novel cycling penalty
    to prevent short-cycling damage to heat pump equipment.
    
    State Space (6D):
        [Indoor_Temp, Outdoor_Temp, Solar_Rad, Price, Last_Action, Time_Index]
    
    Action Space (Discrete 2):
        0 = Heat Pump OFF
        1 = Heat Pump ON
    
    Reward Function:
        R = -(w1*Cost + w2*Discomfort + w3*Cycling_Penalty)
        
        The cycling penalty is the KEY INNOVATION for hardware preservation,
        penalizing switches more frequent than once per 15 minutes.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, data: Optional[pd.DataFrame] = None, 
                 episode_length: int = 1440,  # 24 hours at 1-min resolution
                 comfort_temp_range: Tuple[float, float] = (20.0, 24.0),
                 min_cycle_time: int = 15):  # Minimum 15 minutes between switches
        """
        Initialize the Smart Home Environment.
        
        Args:
            data: AMPds2 DataFrame (if None, generates mock data)
            episode_length: Number of timesteps per episode (minutes)
            comfort_temp_range: (T_min, T_max) for thermal comfort bounds (°C)
            min_cycle_time: Minimum time between HVAC state changes (minutes)
        """
        super().__init__()
        
        # Load or generate data
        if data is None:
            self.data = load_ampds2_mock_data()
        else:
            self.data = data
        
        self.episode_length = episode_length
        self.comfort_temp_min, self.comfort_temp_max = comfort_temp_range
        self.min_cycle_time = min_cycle_time  # Critical for equipment longevity
        
        # RC Thermal Model Parameters (typical for residential buildings)
        # Source: DOE Building America Building Energy Simulation
        self.R = 2.5  # Thermal resistance (°C/kW) - insulation quality
        self.C = 10.0  # Thermal capacitance (kWh/°C) - building thermal mass
        self.dt = 1/60  # 1 minute in hours
        
        # HVAC System Parameters
        self.hvac_power = 3.5  # Heat pump power (kW) - COP included
        self.hvac_cop = 3.0  # Coefficient of Performance
        self.solar_gain_factor = 0.002  # Solar heat gain coefficient (kW per W/m²)
        
        # Reward function weights (tuned for multi-objective optimization)
        self.w_cost = 1.0
        self.w_discomfort = 5.0  # Higher weight for comfort
        self.w_cycling = 10.0  # HIGH PENALTY for equipment wear
        
        # Define Gym spaces
        # State: [T_in, T_out, Solar, Price, Last_Action, Time_of_Day]
        self.observation_space = spaces.Box(
            low=np.array([15.0, -20.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([30.0, 40.0, 1000.0, 0.25, 1.0, 23.99]),
            dtype=np.float32
        )
        
        # Action: [OFF, ON]
        self.action_space = spaces.Discrete(2)
        
        # Episode tracking
        self.current_step = 0
        self.episode_start_idx = 0
        self.indoor_temp = 22.0  # Initial indoor temperature
        self.last_action = 0
        self.time_since_last_switch = self.min_cycle_time  # Start ready to switch
        
        # Metrics for analysis
        self.episode_cost = 0.0
        self.episode_discomfort = 0.0
        self.episode_switches = 0
        self.action_history = []
        self.temp_history = []
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Random start point in dataset
        max_start = len(self.data) - self.episode_length
        self.episode_start_idx = np.random.randint(0, max_start)
        self.current_step = 0
        
        # Reset physical state
        self.indoor_temp = np.random.uniform(21.0, 23.0)  # Random comfortable start
        self.last_action = 0
        self.time_since_last_switch = self.min_cycle_time
        
        # Reset metrics
        self.episode_cost = 0.0
        self.episode_discomfort = 0.0
        self.episode_switches = 0
        self.action_history = []
        self.temp_history = []
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _get_observation(self) -> np.ndarray:
        """Construct the state observation vector."""
        idx = self.episode_start_idx + self.current_step
        row = self.data.iloc[idx]
        
        # Extract time of day (0-23.99)
        timestamp = row['timestamp']
        time_of_day = timestamp.hour + timestamp.minute / 60.0
        
        obs = np.array([
            self.indoor_temp,
            row['Outdoor_Temp'],
            row['Solar_Rad'],
            row['Price'],
            float(self.last_action),
            time_of_day
        ], dtype=np.float32)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Return auxiliary information for analysis."""
        return {
            'indoor_temp': self.indoor_temp,
            'episode_cost': self.episode_cost,
            'episode_discomfort': self.episode_discomfort,
            'episode_switches': self.episode_switches,
            'time_since_last_switch': self.time_since_last_switch
        }
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one timestep within the environment.
        
        This implements the physics-informed dynamics and reward calculation.
        
        Args:
            action: 0 (OFF) or 1 (ON)
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        idx = self.episode_start_idx + self.current_step
        row = self.data.iloc[idx]
        
        # ==================================================================
        # PHYSICS ENGINE: First-Order RC Thermal Model
        # ==================================================================
        # Heat transfer equation:
        # T_in(t+1) = T_in(t) + Δt * [ΔT_conduction + ΔT_hvac + ΔT_solar] / C
        #
        # Where:
        # - ΔT_conduction = (T_out - T_in) / R  (heat loss through envelope)
        # - ΔT_hvac = Q_hvac * COP  (heat pump heating/cooling)
        # - ΔT_solar = Solar_Rad * gain_factor  (passive solar gain)
        # ==================================================================
        
        outdoor_temp = row['Outdoor_Temp']
        solar_rad = row['Solar_Rad']
        
        # Heat flows (kW)
        q_conduction = (outdoor_temp - self.indoor_temp) / self.R
        q_solar = solar_rad * self.solar_gain_factor
        q_hvac = action * self.hvac_power * self.hvac_cop  # Only if ON
        
        # Update indoor temperature using forward Euler integration
        total_heat_flow = q_conduction + q_hvac + q_solar
        self.indoor_temp += self.dt * (total_heat_flow / self.C)
        
        # ==================================================================
        # REWARD FUNCTION: Multi-Objective Optimization
        # ==================================================================
        
        # Component 1: Energy Cost ($/hour)
        energy_consumed = action * self.hvac_power * self.dt  # kWh
        cost = energy_consumed * row['Price']
        
        # Component 2: Thermal Discomfort (quadratic penalty)
        # Penalize temperatures outside comfort zone
        if self.indoor_temp < self.comfort_temp_min:
            discomfort = (self.comfort_temp_min - self.indoor_temp) ** 2
        elif self.indoor_temp > self.comfort_temp_max:
            discomfort = (self.indoor_temp - self.comfort_temp_max) ** 2
        else:
            discomfort = 0.0
        
        # Component 3: SHORT-CYCLING PENALTY (KEY INNOVATION)
        # -------------------------------------------------------
        # Problem: Frequent on/off switching causes:
        #   1. Compressor wear and reduced lifespan
        #   2. Increased maintenance costs
        #   3. Reduced system efficiency
        #   4. Potential refrigerant leakage
        #
        # Solution: Penalize switches within minimum cycle time window
        # This leverages AMPds2's 1-minute resolution to enforce a
        # hardware-aware constraint that typical DRL approaches ignore.
        # -------------------------------------------------------
        cycling_penalty = 0.0
        action_changed = (action != self.last_action)
        
        if action_changed:
            # If switching too soon (before min_cycle_time minutes elapsed)
            if self.time_since_last_switch < self.min_cycle_time:
                # Exponential penalty: earlier switches penalized more
                cycling_penalty = np.exp((self.min_cycle_time - self.time_since_last_switch) / 5.0)
            
            self.episode_switches += 1
            self.time_since_last_switch = 0  # Reset counter
        else:
            self.time_since_last_switch += 1  # Increment minutes since last switch
        
        # Total reward (negative = cost to minimize)
        reward = -(self.w_cost * cost + 
                   self.w_discomfort * discomfort + 
                   self.w_cycling * cycling_penalty)
        
        # Update episode metrics
        self.episode_cost += cost
        self.episode_discomfort += discomfort
        self.action_history.append(action)
        self.temp_history.append(self.indoor_temp)
        
        # Update state
        self.last_action = action
        self.current_step += 1
        
        # Episode termination
        terminated = False
        truncated = (self.current_step >= self.episode_length)
        
        observation = self._get_observation()
        info = self._get_info()
        info['cost'] = cost
        info['discomfort'] = discomfort
        info['cycling_penalty'] = cycling_penalty
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment state (for debugging)."""
        if self.current_step % 60 == 0:  # Print every hour
            print(f"Step {self.current_step}: T_in={self.indoor_temp:.1f}°C, "
                  f"Action={self.last_action}, Switches={self.episode_switches}")


class BaselineThermostatEnv(SmartHomeEnv):
    """
    Baseline Rule-Based Thermostat (for comparison).
    
    Uses simple on/off control with hysteresis band but no cycling prevention.
    This represents current state-of-the-art without PI-DRL.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setpoint = 22.0
        self.hysteresis = 1.0  # ±1°C deadband
    
    def get_baseline_action(self) -> int:
        """Rule-based thermostat control (no cycling awareness)."""
        if self.indoor_temp < self.setpoint - self.hysteresis:
            return 1  # Turn ON
        elif self.indoor_temp > self.setpoint + self.hysteresis:
            return 0  # Turn OFF
        else:
            return self.last_action  # Maintain current state


if __name__ == "__main__":
    # Test the environment
    print("=" * 70)
    print("Testing Physics-Informed Smart Home Environment")
    print("=" * 70)
    
    env = SmartHomeEnv()
    obs, info = env.reset()
    
    print(f"\nInitial State: {obs}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    
    print("\n" + "=" * 70)
    print("Running 60-minute simulation...")
    print("=" * 70)
    
    for i in range(60):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 15 == 0:  # Print every 15 minutes
            print(f"\nMinute {i}:")
            print(f"  Indoor Temp: {info['indoor_temp']:.2f}°C")
            print(f"  Action: {'ON' if action == 1 else 'OFF'}")
            print(f"  Reward: {reward:.4f}")
            print(f"  Switches: {info['episode_switches']}")
            print(f"  Time Since Last Switch: {info['time_since_last_switch']} min")
        
        if terminated or truncated:
            break
    
    print("\n" + "=" * 70)
    print("Episode Complete!")
    print(f"Total Cost: ${info['episode_cost']:.4f}")
    print(f"Total Discomfort: {info['episode_discomfort']:.2f}")
    print(f"Total Switches: {info['episode_switches']}")
    print("=" * 70)
