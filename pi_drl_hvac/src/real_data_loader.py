"""
Real AMPds2 Data Loader
========================

This module loads the REAL AMPds2 dataset from ZIP or CSV files.

Usage:
    1. Download AMPds2 from Harvard Dataverse
    2. Place ZIP file in data/ampds2.zip
    3. Run: python main.py --use-real-data

Author: CPES Research Lab
"""

import os
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime


class RealAMPds2Loader:
    """
    Loader for real AMPds2 dataset.
    
    Handles:
    - ZIP file extraction
    - CSV file parsing
    - Data preprocessing
    - Weather data integration
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the loader.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.ampds2_dir = self.data_dir / "ampds2"
        self.zip_path = self.data_dir / "ampds2.zip"
        
    def load(self, extract_if_needed: bool = True) -> pd.DataFrame:
        """
        Load AMPds2 data.
        
        Args:
            extract_if_needed: Extract ZIP if CSV files don't exist
            
        Returns:
            DataFrame with AMPds2 data ready for use
        """
        # Check if we need to extract
        if extract_if_needed and not self._csv_files_exist():
            self._extract_zip()
        
        # Load and merge CSV files
        data = self._load_csv_files()
        
        # Add weather data (synthetic if not available)
        data = self._add_weather_data(data)
        
        # Add electricity prices
        data = self._add_electricity_prices(data)
        
        # Final preprocessing
        data = self._preprocess(data)
        
        return data
    
    def _csv_files_exist(self) -> bool:
        """Check if extracted CSV files exist."""
        required_files = ['Electricity_WHE.csv']
        for f in required_files:
            # Check in multiple possible locations
            paths_to_check = [
                self.ampds2_dir / f,
                self.ampds2_dir / "AMPds2" / f,
                self.data_dir / f
            ]
            if not any(p.exists() for p in paths_to_check):
                return False
        return True
    
    def _extract_zip(self):
        """Extract ZIP file."""
        if not self.zip_path.exists():
            # Try alternative names
            alt_names = ['AMPds2.zip', 'ampds2_data.zip', 'data.zip']
            for name in alt_names:
                alt_path = self.data_dir / name
                if alt_path.exists():
                    self.zip_path = alt_path
                    break
            else:
                raise FileNotFoundError(
                    f"ZIP file not found. Please place AMPds2 data at:\n"
                    f"  {self.zip_path}\n"
                    f"Or one of: {alt_names}"
                )
        
        print(f"Extracting {self.zip_path}...")
        self.ampds2_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.ampds2_dir)
        
        print(f"Extracted to {self.ampds2_dir}")
        
        # List extracted files
        for f in self.ampds2_dir.rglob("*.csv"):
            print(f"  Found: {f.name}")
    
    def _find_csv(self, name: str) -> Optional[Path]:
        """Find a CSV file in the data directory."""
        # Search in multiple locations
        search_paths = [
            self.ampds2_dir,
            self.ampds2_dir / "AMPds2",
            self.data_dir
        ]
        
        for base in search_paths:
            for f in base.rglob(f"*{name}*"):
                if f.suffix.lower() == '.csv':
                    return f
        return None
    
    def _load_csv_files(self) -> pd.DataFrame:
        """Load and merge CSV files."""
        print("Loading AMPds2 CSV files...")
        
        # Find WHE (Whole House Energy) - required
        whe_path = self._find_csv("WHE")
        if whe_path is None:
            raise FileNotFoundError("Electricity_WHE.csv not found!")
        
        print(f"  Loading: {whe_path.name}")
        whe = pd.read_csv(whe_path)
        
        # Identify timestamp column
        ts_col = None
        for col in ['timestamp', 'unix_ts', 'time', 'datetime', 'UNIX_TS']:
            if col in whe.columns or col.lower() in [c.lower() for c in whe.columns]:
                ts_col = col
                break
        
        if ts_col is None:
            # Assume first column is timestamp
            ts_col = whe.columns[0]
        
        # Convert timestamp
        if whe[ts_col].dtype in ['int64', 'float64']:
            # Unix timestamp
            whe['timestamp'] = pd.to_datetime(whe[ts_col], unit='s')
        else:
            whe['timestamp'] = pd.to_datetime(whe[ts_col])
        
        # Get power column
        power_col = [c for c in whe.columns if c not in [ts_col, 'timestamp']][0]
        data = pd.DataFrame({
            'timestamp': whe['timestamp'],
            'WHE': whe[power_col]
        })
        
        # Try to load HPE (Heat Pump Energy)
        hpe_path = self._find_csv("HPE")
        if hpe_path:
            print(f"  Loading: {hpe_path.name}")
            hpe = pd.read_csv(hpe_path)
            power_col = [c for c in hpe.columns if c.lower() not in ['timestamp', 'unix_ts', 'time']][0]
            data['HPE'] = hpe[power_col].values[:len(data)]
        else:
            print("  HPE not found, estimating from WHE...")
            data['HPE'] = data['WHE'] * 0.4  # Estimate
        
        # Try to load FRE (Furnace Energy)
        fre_path = self._find_csv("FRE")
        if fre_path:
            print(f"  Loading: {fre_path.name}")
            fre = pd.read_csv(fre_path)
            power_col = [c for c in fre.columns if c.lower() not in ['timestamp', 'unix_ts', 'time']][0]
            data['FRE'] = fre[power_col].values[:len(data)]
        else:
            data['FRE'] = 0
        
        print(f"  Loaded {len(data):,} records")
        return data
    
    def _add_weather_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add weather data (real or synthetic)."""
        print("Adding weather data...")
        
        # Try to find weather file
        weather_path = self._find_csv("Weather") or self._find_csv("Climate")
        
        if weather_path:
            print(f"  Loading: {weather_path.name}")
            weather = pd.read_csv(weather_path)
            # TODO: Merge weather data based on timestamp
            # For now, use synthetic
        
        # Generate synthetic weather matching timestamps
        print("  Generating synthetic weather data...")
        n = len(data)
        timestamps = data['timestamp']
        
        # Extract time features
        hours = timestamps.dt.hour + timestamps.dt.minute / 60
        day_of_year = timestamps.dt.dayofyear
        
        # Outdoor temperature (°C) - Canadian climate
        seasonal_temp = -10 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        daily_temp = 5 * np.sin(2 * np.pi * (hours - 9) / 24)
        noise_temp = np.random.normal(0, 2, n)
        data['Outdoor_Temp'] = seasonal_temp + daily_temp + noise_temp
        
        # Solar radiation (W/m²)
        sunrise = 6 - 2 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        sunset = 18 + 2 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        solar_peak = 800 + 200 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        solar = np.zeros(n)
        for i in range(n):
            h = hours.iloc[i]
            if sunrise.iloc[i] < h < sunset.iloc[i]:
                solar_noon = (sunrise.iloc[i] + sunset.iloc[i]) / 2
                day_len = sunset.iloc[i] - sunrise.iloc[i]
                solar[i] = solar_peak.iloc[i] * np.exp(-((h - solar_noon) ** 2) / (day_len ** 2 / 4))
        
        # Cloud cover
        cloud_factor = np.random.uniform(0.3, 1.0, n)
        data['Solar_Radiation'] = np.maximum(solar * cloud_factor, 0)
        
        return data
    
    def _add_electricity_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add Time-of-Use electricity prices."""
        print("Adding electricity prices (Ontario TOU)...")
        
        hours = data['timestamp'].dt.hour + data['timestamp'].dt.minute / 60
        
        prices = np.zeros(len(data))
        for i, h in enumerate(hours):
            if 0 <= h < 7 or h >= 19:
                prices[i] = 0.08  # Off-peak
            elif 7 <= h < 11 or 17 <= h < 19:
                prices[i] = 0.12  # Mid-peak
            else:
                prices[i] = 0.18  # On-peak
        
        # Add small noise
        prices += np.random.normal(0, 0.005, len(data))
        data['Electricity_Price'] = np.maximum(prices, 0.05)
        
        return data
    
    def _preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Final preprocessing steps."""
        print("Preprocessing data...")
        
        # Add time features
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_year'] = data['timestamp'].dt.dayofyear
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['month'] = data['timestamp'].dt.month
        
        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Remove outliers (clip to reasonable ranges)
        data['WHE'] = data['WHE'].clip(0, 20000)
        data['HPE'] = data['HPE'].clip(0, 10000)
        data['Outdoor_Temp'] = data['Outdoor_Temp'].clip(-40, 45)
        
        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        print(f"  Final dataset: {len(data):,} records")
        print(f"  Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        
        return data


def load_real_ampds2(data_dir: str = "data", n_days: Optional[int] = None) -> pd.DataFrame:
    """
    Convenience function to load real AMPds2 data.
    
    Args:
        data_dir: Directory containing data files
        n_days: Limit to first N days (None for all)
        
    Returns:
        DataFrame with AMPds2 data
    """
    loader = RealAMPds2Loader(data_dir=data_dir)
    data = loader.load()
    
    if n_days is not None:
        # Limit to first N days
        start = data['timestamp'].min()
        end = start + pd.Timedelta(days=n_days)
        data = data[data['timestamp'] <= end]
    
    return data


def check_data_availability(data_dir: str = "data") -> dict:
    """
    Check what data files are available.
    
    Args:
        data_dir: Directory to check
        
    Returns:
        Dictionary with availability status
    """
    data_path = Path(data_dir)
    
    status = {
        'zip_file': None,
        'csv_files': [],
        'ready': False
    }
    
    # Check for ZIP
    for name in ['ampds2.zip', 'AMPds2.zip', 'data.zip']:
        if (data_path / name).exists():
            status['zip_file'] = str(data_path / name)
            break
    
    # Check for CSV files
    ampds2_path = data_path / "ampds2"
    if ampds2_path.exists():
        for f in ampds2_path.rglob("*.csv"):
            status['csv_files'].append(f.name)
    
    # Also check data_dir directly
    for f in data_path.glob("*.csv"):
        if f.name not in status['csv_files']:
            status['csv_files'].append(f.name)
    
    # Determine if ready
    status['ready'] = (
        status['zip_file'] is not None or 
        any('WHE' in f for f in status['csv_files'])
    )
    
    return status


if __name__ == "__main__":
    # Check data availability
    print("Checking data availability...")
    status = check_data_availability()
    
    print(f"\nStatus:")
    print(f"  ZIP file: {status['zip_file'] or 'Not found'}")
    print(f"  CSV files: {status['csv_files'] or 'None'}")
    print(f"  Ready: {status['ready']}")
    
    if status['ready']:
        print("\nLoading data...")
        data = load_real_ampds2(n_days=7)
        print(f"\nLoaded {len(data):,} records")
        print(data.head())
    else:
        print("\n⚠️ No data found!")
        print("Please place AMPds2 ZIP file in: data/ampds2.zip")
