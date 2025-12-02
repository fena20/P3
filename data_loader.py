"""
Data Loader Module for AMPds2 Dataset
Generates synthetic data mimicking AMPds2 patterns for testing purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_synthetic_ampds2_data(start_date='2012-04-01', days=365, resolution_minutes=1):
    """
    Generate synthetic AMPds2-like data with realistic patterns.
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    days : int
        Number of days to generate
    resolution_minutes : int
        Data resolution in minutes (AMPds2 uses 1-minute)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: ['WHE', 'HPE', 'FRE', 'Outdoor_Temp', 'Solar_Rad', 'Price', 'timestamp']
    """
    # Generate timestamps
    start = datetime.strptime(start_date, '%Y-%m-%d')
    timestamps = [start + timedelta(minutes=i*resolution_minutes) 
                  for i in range(days * 24 * 60 // resolution_minutes)]
    
    n_samples = len(timestamps)
    
    # Generate outdoor temperature (sinusoidal with daily and seasonal patterns)
    day_of_year = np.array([(t - start).days for t in timestamps])
    hour_of_day = np.array([t.hour + t.minute/60.0 for t in timestamps])
    
    # Seasonal variation (colder in winter, warmer in summer)
    seasonal_temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365.25 - np.pi/2)
    # Daily variation (colder at night, warmer during day)
    daily_temp = 5 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi/2)
    # Add noise
    noise = np.random.normal(0, 2, n_samples)
    outdoor_temp = seasonal_temp + daily_temp + noise
    
    # Solar radiation (zero at night, peak at noon)
    solar_rad = np.maximum(0, 800 * np.sin(np.pi * (hour_of_day - 6) / 12))
    solar_rad[hour_of_day < 6] = 0
    solar_rad[hour_of_day > 18] = 0
    solar_rad += np.random.normal(0, 50, n_samples)
    solar_rad = np.maximum(0, solar_rad)
    
    # Electricity price (Time-of-Use pricing: higher during peak hours 17:00-20:00)
    base_price = 0.10  # $/kWh base rate
    peak_multiplier = np.ones(n_samples)
    peak_hours = (hour_of_day >= 17) & (hour_of_day < 20)
    peak_multiplier[peak_hours] = 1.5  # 50% higher during peak
    price = base_price * peak_multiplier + np.random.normal(0, 0.01, n_samples)
    price = np.maximum(0.05, price)  # Minimum price floor
    
    # Water Heater Energy (WHE) - random spikes throughout day
    whe = np.random.exponential(0.5, n_samples)
    whe_spikes = np.random.random(n_samples) < 0.02  # 2% chance of spike
    whe[whe_spikes] += np.random.exponential(2.0, np.sum(whe_spikes))
    
    # Heat Pump Energy (HPE) - will be controlled by agent, but baseline exists
    # Baseline: ON when outdoor_temp < 18°C (heating mode)
    baseline_hpe = np.where(outdoor_temp < 18, 
                           np.random.normal(2.5, 0.5, n_samples),
                           np.random.normal(0.1, 0.05, n_samples))
    baseline_hpe = np.maximum(0, baseline_hpe)
    
    # Fridge Energy (FRE) - relatively constant with small variations
    fre = np.random.normal(0.15, 0.03, n_samples)
    fre = np.maximum(0.05, fre)
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps,
        'WHE': whe,
        'HPE': baseline_hpe,
        'FRE': fre,
        'Outdoor_Temp': outdoor_temp,
        'Solar_Rad': solar_rad,
        'Price': price
    })
    
    return data


def load_ampds2_data(filepath=None):
    """
    Load AMPds2 data from file or generate synthetic data if file doesn't exist.
    
    Parameters:
    -----------
    filepath : str, optional
        Path to AMPds2 CSV file. If None, generates synthetic data.
    
    Returns:
    --------
    pd.DataFrame
        Processed AMPds2 data
    """
    if filepath and pd.io.common.file_exists(filepath):
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
    else:
        print("No file provided or file not found. Generating synthetic AMPds2-like data...")
        df = generate_synthetic_ampds2_data()
    
    # Ensure required columns exist
    required_cols = ['WHE', 'HPE', 'FRE', 'Outdoor_Temp', 'Solar_Rad', 'Price']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return df
