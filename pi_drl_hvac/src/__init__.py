"""
Physics-Informed Deep Reinforcement Learning (PI-DRL) Framework
for Residential Building HVAC Optimization

Author: Cyber-Physical Energy Systems Research Lab
Target Journal: Applied Energy (Q1)

This package implements a novel PI-DRL approach that integrates:
1. First-order RC thermal model (physics constraints)
2. Cycling penalty for equipment protection
3. Multi-objective reward shaping for demand response

Reference Dataset: AMPds2 (Almanac of Minutely Power Dataset, 2nd Edition)
"""

from .data_loader import AMPds2DataLoader, SyntheticDataGenerator
from .environment import SmartHomeEnv
from .agent import PI_DRL_Agent
from .visualizer import ResultVisualizer

__version__ = "1.0.0"
__author__ = "CPES Research Lab"
