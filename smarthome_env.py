"""
Physics-Informed Smart Home Environment for Deep Reinforcement Learning
Implements RC thermal model with cycling penalty to prevent short-cycling.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from data_loader import load_ampds2_data


class SmartHomeEnv(gym.Env):
    """
    Physics-Informed Smart Home Environment for HVAC Control
    
    State Space (Box(6,)):
        [Indoor_Temp, Outdoor_Temp, Solar_Rad, Price, Last_Action, Time_Index]
    
    Action Space (Discrete(2)):
        0: OFF (Heat Pump OFF)
        1: ON (Heat Pump ON)
    
    Physics Model:
        First-order RC thermal model:
        T_in^{t+1} = T_in^t + Δt × [(T_out - T_in)/R + (Q_HVAC + Q_Solar)/C]
    
    Reward Function:
        R = -(w1·Cost + w2·Discomfort + w3·Cycling_Penalty)
        
    Cycling Penalty:
        Prevents switching more than once every 15 minutes to avoid hardware degradation.
        This addresses the critical "short-cycling" issue in heat pumps.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, data_path=None, 
                 T_setpoint=22.0,  # Comfort setpoint (°C)
                 T_tolerance=1.5,  # Comfort tolerance (°C)
                 R=0.05,  # Thermal resistance (K/kW)
                 C=0.5,  # Thermal capacitance (kWh/K)
                 Q_hvac_max=3.0,  # Maximum HVAC power (kW)
                 dt=1.0/60.0,  # Time step (hours) - 1 minute
                 w1=1.0,  # Cost weight
                 w2=10.0,  # Discomfort weight
                 w3=5.0,  # Cycling penalty weight
                 min_cycle_time=15,  # Minimum cycle time in minutes
                 initial_temp=20.0):  # Initial indoor temperature
        
        super(SmartHomeEnv, self).__init__()
        
        # Load data
        self.data = load_ampds2_data(data_path)
        self.n_steps = len(self.data)
        
        # Physics parameters
        self.T_setpoint = T_setpoint
        self.T_tolerance = T_tolerance
        self.R = R  # Thermal resistance
        self.C = C  # Thermal capacitance
        self.Q_hvac_max = Q_hvac_max
        self.dt = dt  # Time step in hours (1 minute = 1/60 hours)
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.min_cycle_time = min_cycle_time  # Minimum time between state changes (minutes)
        self.initial_temp = initial_temp
        
        # State space: [Indoor_Temp, Outdoor_Temp, Solar_Rad, Price, Last_Action, Time_Index]
        self.observation_space = spaces.Box(
            low=np.array([10.0, -10.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([30.0, 40.0, 1000.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: 0 = OFF, 1 = ON
        self.action_space = spaces.Discrete(2)
        
        # Environment state
        self.current_step = 0
        self.indoor_temp = initial_temp
        self.last_action = 0
        self.last_action_time = -min_cycle_time  # Initialize to allow first action
        self.action_history = []  # Track actions for cycling penalty
        self.time_since_last_switch = min_cycle_time  # Track time since last switch
        
        # Performance tracking
        self.total_cost = 0.0
        self.total_discomfort = 0.0
        self.total_cycles = 0
        self.episode_actions = []
        self.episode_states = []
        self.episode_rewards = []
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.indoor_temp = self.initial_temp
        self.last_action = 0
        self.last_action_time = -self.min_cycle_time
        self.time_since_last_switch = self.min_cycle_time
        
        # Reset tracking
        self.total_cost = 0.0
        self.total_discomfort = 0.0
        self.total_cycles = 0
        self.episode_actions = []
        self.episode_states = []
        self.episode_rewards = []
        
        # Get initial observation
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def _get_observation(self):
        """Construct observation vector from current state"""
        row = self.data.iloc[self.current_step]
        
        # Normalize time index to [0, 1] for better learning
        time_index = self.current_step / self.n_steps
        
        obs = np.array([
            self.indoor_temp / 30.0,  # Normalized indoor temp
            row['Outdoor_Temp'] / 40.0,  # Normalized outdoor temp
            row['Solar_Rad'] / 1000.0,  # Normalized solar radiation
            row['Price'] / 1.0,  # Normalized price
            float(self.last_action),  # Last action (0 or 1)
            time_index  # Normalized time index
        ], dtype=np.float32)
        
        return obs
    
    def _calculate_discomfort(self, temp):
        """
        Calculate discomfort penalty based on deviation from setpoint
        
        Discomfort = |T_in - T_setpoint| - T_tolerance (if outside tolerance)
        """
        deviation = abs(temp - self.T_setpoint)
        if deviation <= self.T_tolerance:
            return 0.0
        else:
            return deviation - self.T_tolerance
    
    def _calculate_cycling_penalty(self, action):
        """
        Calculate cycling penalty to prevent short-cycling
        
        CRITICAL: This function addresses the "short-cycling" problem in heat pumps.
        Short-cycling occurs when a heat pump turns on and off too frequently,
        causing:
        1. Increased wear on compressor and electrical components
        2. Reduced efficiency due to startup transients
        3. Higher energy consumption
        4. Potential equipment failure
        
        The penalty enforces a minimum cycle time (15 minutes) between state changes.
        If the agent switches states within this window, a significant penalty is applied.
        """
        # Check if action represents a state change
        if action != self.last_action:
            # Calculate time since last switch (in minutes)
            time_since_switch = self.current_step - self.last_action_time
            
            if time_since_switch < self.min_cycle_time:
                # Penalty increases as the violation gets worse
                # Maximum penalty when switching immediately after previous switch
                violation_ratio = (self.min_cycle_time - time_since_switch) / self.min_cycle_time
                penalty = violation_ratio * 10.0  # Scale factor for penalty
                return penalty
            else:
                # Valid switch - update tracking
                self.last_action_time = self.current_step
                self.total_cycles += 1
                return 0.0
        else:
            # No state change - no penalty
            return 0.0
    
    def step(self, action):
        """
        Execute one time step in the environment
        
        Implements the physics-informed RC thermal model:
        T_in^{t+1} = T_in^t + Δt × [(T_out - T_in)/R + (Q_HVAC + Q_Solar)/C]
        """
        # Validate action
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        row = self.data.iloc[self.current_step]
        outdoor_temp = row['Outdoor_Temp']
        solar_rad = row['Solar_Rad']
        price = row['Price']
        
        # Calculate HVAC power based on action
        if action == 1:  # ON
            Q_hvac = self.Q_hvac_max
        else:  # OFF
            Q_hvac = 0.0
        
        # Convert solar radiation to heat gain (W/m² to kW, simplified)
        # Assuming effective area and conversion efficiency
        Q_solar = solar_rad * 0.001 * 0.3  # Simplified conversion (kW)
        
        # Physics-Informed Thermal Model (RC Circuit Model)
        # T_in^{t+1} = T_in^t + Δt × [(T_out - T_in)/R + (Q_HVAC + Q_Solar)/C]
        temp_change = self.dt * (
            (outdoor_temp - self.indoor_temp) / self.R + 
            (Q_hvac + Q_solar) / self.C
        )
        self.indoor_temp += temp_change
        
        # Calculate energy cost
        energy_cost = Q_hvac * self.dt * price  # kWh × $/kWh
        
        # Calculate discomfort penalty
        discomfort = self._calculate_discomfort(self.indoor_temp)
        
        # Calculate cycling penalty (prevents short-cycling)
        cycling_penalty = self._calculate_cycling_penalty(action)
        
        # Reward function: R = -(w1·Cost + w2·Discomfort + w3·Cycling_Penalty)
        reward = -(self.w1 * energy_cost + 
                   self.w2 * discomfort + 
                   self.w3 * cycling_penalty)
        
        # Update tracking
        self.total_cost += energy_cost
        self.total_discomfort += discomfort
        self.last_action = action
        self.time_since_last_switch = self.current_step - self.last_action_time
        
        # Store episode data for visualization
        self.episode_actions.append(action)
        self.episode_states.append({
            'indoor_temp': self.indoor_temp,
            'outdoor_temp': outdoor_temp,
            'solar_rad': solar_rad,
            'price': price,
            'time': self.current_step
        })
        self.episode_rewards.append(reward)
        
        # Check termination
        self.current_step += 1
        terminated = self.current_step >= self.n_steps
        truncated = False
        
        # Info dictionary
        info = {
            'cost': energy_cost,
            'discomfort': discomfort,
            'cycling_penalty': cycling_penalty,
            'indoor_temp': self.indoor_temp,
            'total_cost': self.total_cost,
            'total_discomfort': self.total_discomfort,
            'total_cycles': self.total_cycles
        }
        
        # Get next observation
        if terminated:
            obs = self._get_observation()  # Final observation
        else:
            obs = self._get_observation()
        
        return obs, reward, terminated, truncated, info
    
    def get_episode_data(self):
        """Return episode data for visualization"""
        return {
            'actions': self.episode_actions,
            'states': self.episode_states,
            'rewards': self.episode_rewards,
            'total_cost': self.total_cost,
            'total_discomfort': self.total_discomfort,
            'total_cycles': self.total_cycles
        }
