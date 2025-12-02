"""
Physics-Informed Smart Home Environment
========================================

This module implements the core contribution of our PI-DRL framework:
a Gymnasium environment that embeds physics constraints (RC thermal model)
and a novel reward function with cycling penalty for equipment protection.

Key Innovations:
1. First-order RC thermal model for realistic temperature dynamics
2. Cycling penalty leveraging AMPds2's 1-minute resolution
3. Multi-objective reward balancing cost, comfort, and equipment life

Physics Model Reference:
- Bacher, P., & Madsen, H. (2011). Identifying suitable models for the heat 
  dynamics of buildings. Energy and Buildings, 43(7), 1511-1522.

Author: CPES Research Lab
Target: Applied Energy (Q1 Journal)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Optional, Any
import pandas as pd

from .data_loader import SyntheticDataGenerator


class SmartHomeEnv(gym.Env):
    """
    Physics-Informed Smart Home Environment for HVAC Control.
    
    This environment models a residential building with:
    - Heat pump as the primary HVAC actuator
    - 1st-order RC thermal model for temperature dynamics
    - Time-of-use electricity pricing
    - Solar heat gains
    
    State Space (Box[6]):
        [0] Indoor Temperature (°C): Current building temperature
        [1] Outdoor Temperature (°C): Ambient temperature from weather data
        [2] Solar Radiation (W/m²): Solar irradiance
        [3] Electricity Price ($/kWh): Time-of-use rate
        [4] Last Action (0/1): Previous control action
        [5] Time Index (normalized): Hour of day normalized to [0,1]
    
    Action Space (Discrete[2]):
        0: Heat Pump OFF
        1: Heat Pump ON
    
    Reward Function (Novel Multi-Objective):
        R = -(w1*Cost + w2*Discomfort + w3*Cycling_Penalty)
        
        Where Cycling_Penalty addresses the SHORT-CYCLING problem:
        - Heat pump compressors suffer wear from frequent on/off cycles
        - Industry standard: minimum 15-minute run/off time
        - We penalize state changes within 15 minutes of last switch
    
    Attributes:
        R_thermal: Thermal resistance (°C/kW)
        C_thermal: Thermal capacitance (kWh/°C)
        dt: Time step (hours) - 1 minute for AMPds2
        Q_hvac: Heat pump thermal output (kW)
        
    Example:
        >>> env = SmartHomeEnv()
        >>> obs, info = env.reset()
        >>> action = 1  # Turn ON
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    # =========================================================================
    # THERMAL MODEL PARAMETERS (Calibrated for typical residential building)
    # =========================================================================
    # These values represent a well-insulated single-family home (~150 m²)
    
    # Thermal Resistance: R = 5 °C/kW (represents wall + window + infiltration)
    # Higher R = better insulation
    DEFAULT_R_THERMAL = 5.0  # °C/kW
    
    # Thermal Capacitance: C = 10 kWh/°C (building thermal mass)
    # Higher C = more thermal inertia, slower temperature changes
    DEFAULT_C_THERMAL = 10.0  # kWh/°C
    
    # Heat Pump Parameters
    HEAT_PUMP_POWER = 3.0  # kW electrical input
    HEAT_PUMP_COP = 3.5    # Coefficient of Performance
    
    # Solar Heat Gain Coefficient (SHGC * Window Area * Transmittance)
    SOLAR_GAIN_FACTOR = 0.01  # kW per W/m² of solar radiation
    
    # =========================================================================
    # REWARD FUNCTION WEIGHTS (Tuned for balanced optimization)
    # =========================================================================
    DEFAULT_WEIGHTS = {
        'cost': 1.0,           # w1: Energy cost weight
        'comfort': 2.0,        # w2: Comfort violation weight (prioritize comfort)
        'cycling': 0.5         # w3: Cycling penalty weight
    }
    
    # Comfort Parameters
    TEMP_SETPOINT = 21.0       # °C (target indoor temperature)
    TEMP_DEADBAND = 2.0        # °C (acceptable deviation)
    
    # =========================================================================
    # CYCLING PENALTY PARAMETERS
    # =========================================================================
    # Short-cycling definition: State change within MIN_CYCLE_TIME
    # Heat pump compressors require minimum run/off times to:
    # 1. Allow lubricant to properly distribute
    # 2. Prevent excessive start-up current stress
    # 3. Allow refrigerant pressures to equalize
    # Industry standard: 15 minutes minimum
    MIN_CYCLE_TIME = 15  # minutes (critical for equipment protection)
    
    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        episode_length: int = 1440,  # 1 day = 1440 minutes
        R_thermal: float = DEFAULT_R_THERMAL,
        C_thermal: float = DEFAULT_C_THERMAL,
        weights: Optional[Dict[str, float]] = None,
        random_seed: int = 42,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the Smart Home Environment.
        
        Args:
            data: Pre-loaded weather/price data. If None, generates synthetic.
            episode_length: Number of time steps per episode (minutes)
            R_thermal: Thermal resistance (°C/kW)
            C_thermal: Thermal capacitance (kWh/°C)
            weights: Custom reward weights dict {cost, comfort, cycling}
            random_seed: Seed for reproducibility
            render_mode: Rendering mode ('human' or 'rgb_array')
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.np_random = np.random.default_rng(random_seed)
        
        # Thermal model parameters
        self.R_thermal = R_thermal
        self.C_thermal = C_thermal
        self.dt = 1.0 / 60.0  # Time step: 1 minute = 1/60 hour
        
        # HVAC parameters
        self.Q_hvac_max = self.HEAT_PUMP_POWER * self.HEAT_PUMP_COP  # kW thermal
        
        # Reward weights
        self.weights = weights if weights else self.DEFAULT_WEIGHTS.copy()
        
        # Episode parameters
        self.episode_length = episode_length
        
        # Load or generate data
        if data is not None:
            self.data = data.reset_index(drop=True)
        else:
            generator = SyntheticDataGenerator(random_seed=random_seed)
            # Generate enough data for multiple episodes
            self.data = generator.generate(
                start_date="2012-04-01",
                end_date="2012-04-30"  # 30 days
            ).reset_index(drop=True)
        
        # =====================================================================
        # DEFINE SPACES
        # =====================================================================
        
        # State Space: Box(6,)
        # [Indoor_Temp, Outdoor_Temp, Solar_Rad, Price, Last_Action, Time_Index]
        self.observation_space = spaces.Box(
            low=np.array([0, -30, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([40, 45, 1200, 0.5, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action Space: Discrete(2) [OFF, ON]
        self.action_space = spaces.Discrete(2)
        
        # State variables (initialized in reset)
        self._initialize_state_variables()
        
        # Tracking for visualization
        self.history = {
            'indoor_temp': [],
            'outdoor_temp': [],
            'action': [],
            'reward': [],
            'cost': [],
            'comfort': [],
            'cycling_penalty': [],
            'timestamp': [],
            'price': [],
            'solar': []
        }
        
    def _initialize_state_variables(self):
        """Initialize/reset state variables."""
        self.current_step = 0
        self.indoor_temp = self.TEMP_SETPOINT  # Start at setpoint
        self.last_action = 0
        self.time_since_switch = self.MIN_CYCLE_TIME  # Allow initial switch
        self.data_index = 0
        self.total_energy = 0.0
        self.total_cost = 0.0
        self.comfort_violations = 0
        self.cycle_count = 0
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (start_index, initial_temp)
            
        Returns:
            observation: Initial state observation
            info: Additional information dict
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        
        # Reset state
        self._initialize_state_variables()
        
        # Handle options
        if options:
            self.data_index = options.get('start_index', 
                self.np_random.integers(0, len(self.data) - self.episode_length - 1))
            self.indoor_temp = options.get('initial_temp', 
                self.TEMP_SETPOINT + self.np_random.uniform(-2, 2))
        else:
            # Random starting point in data
            max_start = max(0, len(self.data) - self.episode_length - 1)
            self.data_index = self.np_random.integers(0, max_start) if max_start > 0 else 0
            # Random initial temperature within comfort zone
            self.indoor_temp = self.TEMP_SETPOINT + self.np_random.uniform(-1, 1)
        
        # Clear history
        for key in self.history:
            self.history[key].clear()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step with physics-based dynamics.
        
        This is the core of our Physics-Informed approach:
        1. Apply RC thermal model for temperature update
        2. Calculate multi-objective reward with cycling penalty
        
        Args:
            action: HVAC control action (0=OFF, 1=ON)
            
        Returns:
            observation: New state observation
            reward: Scalar reward
            terminated: Whether episode ended naturally
            truncated: Whether episode was cut short
            info: Additional information dict
        """
        # Get current environmental conditions from data
        row = self.data.iloc[self.data_index]
        T_out = row['Outdoor_Temp']
        Q_solar = row['Solar_Radiation'] * self.SOLAR_GAIN_FACTOR  # kW
        price = row['Electricity_Price']
        
        # =====================================================================
        # PHYSICS ENGINE: 1st-Order RC Thermal Model
        # =====================================================================
        # Equation:
        # T_in(t+1) = T_in(t) + dt * [(T_out - T_in)/R + (Q_HVAC + Q_Solar)/C]
        #
        # Where:
        # - T_in: Indoor temperature (°C)
        # - T_out: Outdoor temperature (°C)
        # - R: Thermal resistance (°C/kW)
        # - C: Thermal capacitance (kWh/°C)
        # - Q_HVAC: Heat pump thermal output (kW)
        # - Q_Solar: Solar heat gain (kW)
        # - dt: Time step (hours)
        
        # Calculate HVAC thermal output
        Q_hvac = action * self.Q_hvac_max  # kW (0 if OFF, max if ON)
        
        # Thermal dynamics
        dT_conduction = (T_out - self.indoor_temp) / self.R_thermal  # Heat flow from outside
        dT_sources = (Q_hvac + Q_solar) / self.C_thermal            # Heat from HVAC + solar
        
        # Temperature update (forward Euler integration)
        T_new = self.indoor_temp + self.dt * (dT_conduction + dT_sources)
        
        # =====================================================================
        # REWARD CALCULATION (Novel Multi-Objective)
        # =====================================================================
        
        # --- Component 1: Energy Cost ---
        energy_kwh = action * self.HEAT_PUMP_POWER * self.dt  # kWh consumed
        cost = energy_kwh * price
        
        # --- Component 2: Comfort Violation ---
        temp_deviation = abs(T_new - self.TEMP_SETPOINT)
        if temp_deviation > self.TEMP_DEADBAND:
            discomfort = (temp_deviation - self.TEMP_DEADBAND) ** 2
            self.comfort_violations += 1
        else:
            discomfort = 0.0
        
        # --- Component 3: Cycling Penalty (KEY INNOVATION) ---
        # =====================================================================
        # SHORT-CYCLING PREVENTION
        # =====================================================================
        # Problem: Heat pump compressors suffer mechanical stress from rapid
        # on/off cycling. This causes:
        #   1. Increased wear on compressor motor windings
        #   2. Lubricant displacement issues
        #   3. Liquid slugging risk when pressures don't equalize
        #   4. Higher starting currents causing electrical stress
        #
        # Solution: We leverage AMPds2's 1-minute resolution to track exact
        # time since last state change and penalize switches within 15 minutes.
        #
        # The penalty is exponential to strongly discourage rapid cycling:
        # penalty = exp(-time_since_switch / 5) if switching
        #
        # This creates a learning signal that the agent must:
        # - "Commit" to ON or OFF states for meaningful durations
        # - Pre-cool/pre-heat to avoid needing rapid corrections
        # =====================================================================
        
        cycling_penalty = 0.0
        if action != self.last_action:
            # State changed - check if it's short-cycling
            if self.time_since_switch < self.MIN_CYCLE_TIME:
                # Exponential penalty: Stronger penalty for faster switching
                cycling_penalty = np.exp(-self.time_since_switch / 5.0)
            self.cycle_count += 1
            self.time_since_switch = 0
        
        # Increment time since switch (cap at MIN_CYCLE_TIME for numerical stability)
        self.time_since_switch = min(self.time_since_switch + 1, self.MIN_CYCLE_TIME + 1)
        
        # --- Combine into total reward ---
        reward = -(
            self.weights['cost'] * cost +
            self.weights['comfort'] * discomfort +
            self.weights['cycling'] * cycling_penalty
        )
        
        # =====================================================================
        # STATE UPDATE
        # =====================================================================
        self.indoor_temp = T_new
        self.last_action = action
        self.current_step += 1
        self.data_index += 1
        self.total_energy += energy_kwh
        self.total_cost += cost
        
        # Record history
        self.history['indoor_temp'].append(T_new)
        self.history['outdoor_temp'].append(T_out)
        self.history['action'].append(action)
        self.history['reward'].append(reward)
        self.history['cost'].append(cost)
        self.history['comfort'].append(discomfort)
        self.history['cycling_penalty'].append(cycling_penalty)
        self.history['price'].append(price)
        self.history['solar'].append(Q_solar)
        if 'timestamp' in row.index:
            self.history['timestamp'].append(row['timestamp'])
        
        # Check termination
        terminated = False
        truncated = self.current_step >= self.episode_length
        truncated = truncated or self.data_index >= len(self.data) - 1
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct the observation vector.
        
        Returns:
            obs: [Indoor_Temp, Outdoor_Temp, Solar_Rad, Price, Last_Action, Time_Index]
        """
        row = self.data.iloc[min(self.data_index, len(self.data) - 1)]
        
        # Normalize time index to [0, 1]
        hour = row['hour'] + row.get('minute', 0) / 60 if 'minute' in row else row['hour']
        time_index = hour / 24.0
        
        obs = np.array([
            self.indoor_temp,
            row['Outdoor_Temp'],
            row['Solar_Radiation'],
            row['Electricity_Price'],
            float(self.last_action),
            time_index
        ], dtype=np.float32)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information about the current state.
        
        Returns:
            info: Dictionary with debug/logging information
        """
        return {
            'step': self.current_step,
            'indoor_temp': self.indoor_temp,
            'total_energy_kwh': self.total_energy,
            'total_cost': self.total_cost,
            'comfort_violations': self.comfort_violations,
            'cycle_count': self.cycle_count,
            'time_since_switch': self.time_since_switch
        }
    
    def get_history_df(self) -> pd.DataFrame:
        """
        Get episode history as a DataFrame.
        
        Returns:
            DataFrame with all tracked variables
        """
        return pd.DataFrame(self.history)
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment (optional visualization)."""
        if self.render_mode == "human":
            print(f"Step {self.current_step}: T_in={self.indoor_temp:.1f}°C, "
                  f"Action={self.last_action}, Cost=${self.total_cost:.4f}")
        return None
    
    def close(self):
        """Clean up resources."""
        pass


class BaselineThemostatEnv(SmartHomeEnv):
    """
    Baseline thermostat environment for comparison.
    
    Implements a simple on/off thermostat with hysteresis:
    - Turn ON when T < setpoint - deadband/2
    - Turn OFF when T > setpoint + deadband/2
    
    No cycling penalty is applied, demonstrating the short-cycling problem.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights['cycling'] = 0.0  # Disable cycling penalty
        
    def get_thermostat_action(self) -> int:
        """
        Get the action from a simple thermostat controller.
        
        Returns:
            action: 0 (OFF) or 1 (ON)
        """
        lower_bound = self.TEMP_SETPOINT - self.TEMP_DEADBAND / 2
        upper_bound = self.TEMP_SETPOINT + self.TEMP_DEADBAND / 2
        
        if self.indoor_temp < lower_bound:
            return 1  # Turn ON
        elif self.indoor_temp > upper_bound:
            return 0  # Turn OFF
        else:
            return self.last_action  # Maintain current state


if __name__ == "__main__":
    # Quick test of the environment
    print("Testing SmartHomeEnv...")
    
    env = SmartHomeEnv(episode_length=100)
    obs, info = env.reset()
    
    print(f"Initial observation: {obs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Run a few steps
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        print(f"Step {i+1}: Action={action}, Reward={reward:.4f}, "
              f"T_in={info['indoor_temp']:.1f}°C")
    
    print(f"\nTotal reward over 10 steps: {total_reward:.4f}")
    print(f"Final info: {info}")
