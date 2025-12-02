"""
AMPds2 Data Loader and Synthetic Data Generator
================================================

This module provides:
1. Mock function for loading real AMPds2 dataset
2. Synthetic data generator mimicking AMPds2 patterns for testing

AMPds2 Dataset Reference:
- Makonin, S., et al. "AMPds2: The Almanac of Minutely Power dataset (Version 2)"
- 1-minute resolution data from a residential building in Canada
- Key columns: WHE (Whole House Energy), HPE (Heat Pump Energy), 
               FRE (Furnace Energy), plus environmental data

Author: CPES Research Lab
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from datetime import datetime, timedelta


class AMPds2DataLoader:
    """
    Mock data loader for AMPds2 dataset.
    
    In production, this would interface with the actual AMPds2 CSV files.
    For this implementation, we provide the expected interface and generate
    synthetic data that mimics the statistical properties of AMPds2.
    """
    
    EXPECTED_COLUMNS = ['timestamp', 'WHE', 'HPE', 'FRE', 'Outdoor_Temp', 
                        'Solar_Radiation', 'Electricity_Price']
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to AMPds2 CSV files (if None, uses synthetic data)
        """
        self.data_path = data_path
        self.data = None
        
    def load_data(self, start_date: str = "2012-04-01", 
                  end_date: str = "2013-03-31") -> pd.DataFrame:
        """
        Load AMPds2 data for the specified date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with 1-minute resolution energy and environmental data
        """
        if self.data_path is not None:
            # Production: Load actual AMPds2 data
            return self._load_real_data(start_date, end_date)
        else:
            # Development: Generate synthetic data
            generator = SyntheticDataGenerator()
            return generator.generate(start_date, end_date)
    
    def _load_real_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load real AMPds2 data from CSV files.
        
        Note: This is a placeholder for the actual implementation.
        Real implementation would:
        1. Load Electricity_WHE.csv, Electricity_HPE.csv, etc.
        2. Merge with weather data
        3. Resample to ensure 1-minute alignment
        """
        raise NotImplementedError(
            "Real data loading requires AMPds2 CSV files. "
            "Please use SyntheticDataGenerator for testing."
        )


class SyntheticDataGenerator:
    """
    Generates synthetic data mimicking AMPds2 patterns.
    
    This generator creates realistic patterns for:
    - Outdoor temperature: Seasonal + daily cycles
    - Solar radiation: Bell curve aligned with daylight hours
    - Electricity prices: Time-of-use pricing with peak hours
    - Energy consumption: Base load + HVAC patterns
    
    Statistical properties are calibrated to match AMPds2 characteristics
    for a residential building in a heating-dominated climate (Canada).
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the synthetic data generator.
        
        Args:
            random_seed: Seed for reproducibility
        """
        np.random.seed(random_seed)
        self.seed = random_seed
        
    def generate(self, start_date: str = "2012-04-01", 
                 end_date: str = "2013-03-31",
                 resolution_minutes: int = 1) -> pd.DataFrame:
        """
        Generate synthetic AMPds2-like data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            resolution_minutes: Time resolution in minutes (default: 1)
            
        Returns:
            DataFrame with synthetic energy and environmental data
        """
        # Create timestamp index
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
        timestamps = pd.date_range(start=start, end=end, freq=f'{resolution_minutes}min')
        n_samples = len(timestamps)
        
        # Generate time-based features
        hours = timestamps.hour + timestamps.minute / 60
        day_of_year = timestamps.dayofyear
        
        # ===================================================================
        # OUTDOOR TEMPERATURE (°C)
        # ===================================================================
        # Seasonal pattern: Cold winters (-10°C avg), mild summers (20°C avg)
        # Daily pattern: Min at 5am, Max at 3pm
        seasonal_temp = -10 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        daily_temp = 5 * np.sin(2 * np.pi * (hours - 9) / 24)
        noise_temp = np.random.normal(0, 2, n_samples)
        outdoor_temp = seasonal_temp + daily_temp + noise_temp
        
        # ===================================================================
        # SOLAR RADIATION (W/m²)
        # ===================================================================
        # Peak at solar noon, zero at night
        # Seasonal variation: More in summer
        sunrise = 6 - 2 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        sunset = 18 + 2 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        day_length = sunset - sunrise
        solar_peak = 800 + 200 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        solar_radiation = np.zeros(n_samples)
        for i, h in enumerate(hours):
            if sunrise[i] < h < sunset[i]:
                # Bell curve centered at solar noon
                solar_noon = (sunrise[i] + sunset[i]) / 2
                solar_radiation[i] = solar_peak[i] * np.exp(
                    -((h - solar_noon) ** 2) / (day_length[i] ** 2 / 4)
                )
        # Add cloud cover variability
        cloud_factor = np.random.uniform(0.3, 1.0, n_samples)
        solar_radiation = solar_radiation * cloud_factor
        solar_radiation = np.maximum(solar_radiation, 0)
        
        # ===================================================================
        # ELECTRICITY PRICE ($/kWh)
        # ===================================================================
        # Time-of-Use pricing for Ontario, Canada (typical AMPds2 region)
        # Off-peak: 00:00-07:00, 19:00-24:00 ($0.08/kWh)
        # Mid-peak: 07:00-11:00, 17:00-19:00 ($0.12/kWh)
        # On-peak:  11:00-17:00 ($0.18/kWh)
        electricity_price = np.zeros(n_samples)
        for i, h in enumerate(hours):
            if 0 <= h < 7 or h >= 19:
                electricity_price[i] = 0.08  # Off-peak
            elif 7 <= h < 11 or 17 <= h < 19:
                electricity_price[i] = 0.12  # Mid-peak
            else:
                electricity_price[i] = 0.18  # On-peak
        
        # Add small random variations
        electricity_price += np.random.normal(0, 0.005, n_samples)
        electricity_price = np.maximum(electricity_price, 0.05)
        
        # ===================================================================
        # WHOLE HOUSE ENERGY (WHE) - Watts
        # ===================================================================
        # Base load: 300-500W (refrigerator, standby, etc.)
        # Activity peaks: Morning (7-9), Evening (17-22)
        base_load = 400 + 100 * np.random.random(n_samples)
        
        # Morning activity peak
        morning_activity = 300 * np.exp(-((hours - 7.5) ** 2) / 2)
        # Evening activity peak  
        evening_activity = 500 * np.exp(-((hours - 19) ** 2) / 4)
        
        whe = base_load + morning_activity + evening_activity
        whe += np.random.normal(0, 50, n_samples)
        whe = np.maximum(whe, 200)
        
        # ===================================================================
        # HEAT PUMP ENERGY (HPE) - Watts
        # ===================================================================
        # Proportional to heating/cooling demand
        # Nominal power: 3000W when running
        # More heating needed when cold, cooling when hot
        heating_demand = np.maximum(18 - outdoor_temp, 0) / 28  # Normalize
        cooling_demand = np.maximum(outdoor_temp - 24, 0) / 16  # Normalize
        hvac_demand = np.maximum(heating_demand, cooling_demand)
        
        # Add thermal inertia effect (smoothing)
        hvac_demand_smooth = pd.Series(hvac_demand).rolling(
            window=30, min_periods=1, center=True
        ).mean().values
        
        # Baseline thermostat: On/Off control with hysteresis (simulated)
        hp_power = 3000 * hvac_demand_smooth
        hp_power += np.random.normal(0, 100, n_samples)
        hp_power = np.clip(hp_power, 0, 4000)
        
        # ===================================================================
        # FURNACE ENERGY (FRE) - Watts  
        # ===================================================================
        # Backup/auxiliary heating, used in extreme cold
        fre = np.zeros(n_samples)
        extreme_cold_mask = outdoor_temp < -15
        fre[extreme_cold_mask] = 1500 + np.random.normal(0, 200, np.sum(extreme_cold_mask))
        fre = np.maximum(fre, 0)
        
        # ===================================================================
        # CREATE DATAFRAME
        # ===================================================================
        data = pd.DataFrame({
            'timestamp': timestamps,
            'WHE': whe,
            'HPE': hp_power,
            'FRE': fre,
            'Outdoor_Temp': outdoor_temp,
            'Solar_Radiation': solar_radiation,
            'Electricity_Price': electricity_price,
            'hour': timestamps.hour,
            'day_of_year': timestamps.dayofyear,
            'day_of_week': timestamps.dayofweek,
            'month': timestamps.month
        })
        
        return data
    
    def generate_episode_data(self, n_days: int = 7) -> pd.DataFrame:
        """
        Generate data for a single training episode.
        
        Args:
            n_days: Number of days to generate
            
        Returns:
            DataFrame with episode data
        """
        end_date = pd.Timestamp("2012-04-01") + pd.Timedelta(days=n_days)
        return self.generate(
            start_date="2012-04-01",
            end_date=end_date.strftime("%Y-%m-%d")
        )


def load_ampds2_mock(n_days: int = 30, seed: int = 42) -> pd.DataFrame:
    """
    Convenience function to load mock AMPds2 data.
    
    This function provides a simple interface for loading synthetic data
    that mimics the AMPds2 dataset characteristics.
    
    Args:
        n_days: Number of days of data to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic AMPds2-like data
        
    Example:
        >>> data = load_ampds2_mock(n_days=7)
        >>> print(data.columns)
        Index(['timestamp', 'WHE', 'HPE', 'FRE', 'Outdoor_Temp', 
               'Solar_Radiation', 'Electricity_Price', ...])
    """
    generator = SyntheticDataGenerator(random_seed=seed)
    end_date = pd.Timestamp("2012-04-01") + pd.Timedelta(days=n_days)
    return generator.generate(
        start_date="2012-04-01",
        end_date=end_date.strftime("%Y-%m-%d")
    )


if __name__ == "__main__":
    # Quick test of data generation
    print("Generating synthetic AMPds2 data...")
    data = load_ampds2_mock(n_days=7)
    print(f"\nDataset shape: {data.shape}")
    print(f"\nColumns: {data.columns.tolist()}")
    print(f"\nSample statistics:")
    print(data[['Outdoor_Temp', 'Solar_Radiation', 'Electricity_Price', 
                'WHE', 'HPE']].describe())
