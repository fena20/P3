"""
Test with Real AMPds2 Dataset
Downloads and processes real AMPds2 data for comprehensive testing
"""

import numpy as np
import pandas as pd
import os
import urllib.request
import zipfile
from smarthome_env import SmartHomeEnv
from visualizer import ResultVisualizer
from table_generator import TableGenerator
from generate_tables_demo import (
    run_baseline_thermostat, run_piddrl_with_cycling_penalty, run_drl_no_cycling_penalty
)
from sensitivity_analysis import SensitivityAnalyzer
import warnings
warnings.filterwarnings('ignore')


def download_ampds2_sample():
    """
    Download sample AMPds2 data or provide instructions
    
    Note: Full AMPds2 dataset requires registration at ampds.org
    This function provides a template for loading real data
    """
    print("=" * 80)
    print("AMPds2 Dataset Loading")
    print("=" * 80)
    print()
    print("To use real AMPds2 dataset:")
    print("1. Register and download from: https://ampds.org/")
    print("2. Extract the CSV files")
    print("3. Place in ./data/ampds2/ directory")
    print("4. Ensure columns: timestamp, WHE, HPE, FRE, Outdoor_Temp, Solar_Rad, Price")
    print()
    
    data_dir = "./data/ampds2"
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if data exists
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if csv_files:
        print(f"Found {len(csv_files)} CSV files in {data_dir}/")
        return csv_files[0]  # Use first CSV file found
    else:
        print("No CSV files found. Using synthetic data.")
        print("To use real data, place AMPds2 CSV files in ./data/ampds2/")
        return None


def load_real_ampds2_data(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess real AMPds2 data
    
    Parameters:
    -----------
    filepath : str
        Path to AMPds2 CSV file
    
    Returns:
    --------
    pd.DataFrame
        Processed data with required columns
    """
    print(f"Loading real AMPds2 data from {filepath}...")
    
    try:
        # Try to load CSV
        df = pd.read_csv(filepath, parse_dates=['timestamp'], low_memory=False)
        
        # Check required columns
        required_cols = ['WHE', 'HPE', 'FRE', 'Outdoor_Temp', 'Solar_Rad', 'Price']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")
            print("Attempting to map from common AMPds2 column names...")
            
            # Common AMPds2 column name mappings
            column_mapping = {
                'WHE': ['WHE', 'WaterHeater', 'WH', 'water_heater'],
                'HPE': ['HPE', 'HeatPump', 'HP', 'heat_pump'],
                'FRE': ['FRE', 'Fridge', 'FR', 'fridge', 'Refrigerator'],
                'Outdoor_Temp': ['Outdoor_Temp', 'OutTemp', 'T_out', 'outdoor_temp', 'Temperature'],
                'Solar_Rad': ['Solar_Rad', 'Solar', 'SolarRad', 'solar_rad', 'Irradiance'],
                'Price': ['Price', 'price', 'ElectricityPrice', 'TOU']
            }
            
            # Try to find and map columns
            for target_col, possible_names in column_mapping.items():
                if target_col not in df.columns:
                    for name in possible_names:
                        if name in df.columns:
                            df[target_col] = df[name]
                            print(f"  Mapped {name} -> {target_col}")
                            break
            
            # Check again
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Cannot find columns: {missing_cols}")
        
        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            if 'Date' in df.columns and 'Time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            elif 'DateTime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['DateTime'])
            else:
                raise ValueError("Cannot find timestamp column")
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure numeric columns
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add price if missing (create Time-of-Use pricing)
        if 'Price' not in df.columns or df['Price'].isna().all():
            print("  Generating Time-of-Use pricing...")
            hour = df['timestamp'].dt.hour
            df['Price'] = 0.10  # Base price
            df.loc[(hour >= 17) & (hour < 20), 'Price'] = 0.15  # Peak price
        
        print(f"✓ Loaded {len(df)} samples")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Columns: {list(df.columns)}")
        
        return df[['timestamp'] + required_cols]
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Falling back to synthetic data...")
        return None


def test_with_real_data(data_path: str = None, n_samples: int = 5000):
    """
    Test PI-DRL framework with real AMPds2 data
    
    Parameters:
    -----------
    data_path : str
        Path to real AMPds2 CSV file
    n_samples : int
        Number of samples to use for testing
    """
    print("=" * 80)
    print("TESTING WITH REAL AMPDS2 DATASET")
    print("=" * 80)
    print()
    
    # Try to load real data
    if data_path and os.path.exists(data_path):
        real_data = load_real_ampds2_data(data_path)
        if real_data is not None:
            print("✓ Using REAL AMPds2 data")
            # Save processed data for environment
            processed_path = "./data/ampds2/processed_data.csv"
            real_data.to_csv(processed_path, index=False)
            data_path = processed_path
        else:
            print("⚠ Falling back to synthetic data")
            data_path = None
    else:
        # Try to find data in default location
        default_path = "./data/ampds2"
        if os.path.exists(default_path):
            csv_files = [f for f in os.listdir(default_path) if f.endswith('.csv')]
            if csv_files:
                data_path = os.path.join(default_path, csv_files[0])
                real_data = load_real_ampds2_data(data_path)
                if real_data is not None:
                    processed_path = "./data/ampds2/processed_data.csv"
                    real_data.to_csv(processed_path, index=False)
                    data_path = processed_path
                else:
                    data_path = None
        
        if data_path is None:
            print("⚠ No real data found. Using synthetic data.")
            print("  To use real AMPds2 data:")
            print("  1. Download from https://ampds.org/")
            print("  2. Place CSV file in ./data/ampds2/")
            print("  3. Ensure columns: timestamp, WHE, HPE, FRE, Outdoor_Temp, Solar_Rad, Price")
            data_path = None
    
    # Create environments with real or synthetic data
    print("\nStep 1: Creating environments...")
    baseline_env = SmartHomeEnv(data_path=data_path)
    piddrl_env = SmartHomeEnv(data_path=data_path)
    drl_no_penalty_env = SmartHomeEnv(data_path=data_path)
    
    # Limit samples if using real data
    if data_path and len(baseline_env.data) < n_samples:
        n_samples = len(baseline_env.data)
        print(f"  Using {n_samples} samples from real data")
    
    print("✓ Environments created")
    print()
    
    # Run simulations
    print("Step 2: Running simulations...")
    print("  → Baseline thermostat...")
    baseline_metrics = run_baseline_thermostat(baseline_env, n_samples=n_samples)
    
    print("  → PI-DRL with cycling penalty...")
    piddrl_metrics = run_piddrl_with_cycling_penalty(piddrl_env, n_samples=n_samples)
    
    print("  → DRL without cycling penalty...")
    drl_no_penalty_metrics = run_drl_no_cycling_penalty(drl_no_penalty_env, n_samples=n_samples)
    
    print("✓ Simulations completed")
    print()
    
    # Display results
    print("=" * 80)
    print("RESULTS WITH REAL DATA")
    print("=" * 80)
    print()
    print("Baseline Thermostat:")
    print(f"  Cost: ${baseline_metrics['cost']:.2f}")
    print(f"  Discomfort: {baseline_metrics['comfort']:.2f} degree-hours")
    print(f"  Cycles: {baseline_metrics['cycles']:.0f}")
    print(f"  Avg Cycle Duration: {baseline_metrics.get('avg_cycle_duration', 0):.1f} min")
    print()
    
    print("PI-DRL (With Cycling Penalty):")
    print(f"  Cost: ${piddrl_metrics['cost']:.2f}")
    print(f"  Discomfort: {piddrl_metrics['comfort']:.2f} degree-hours")
    print(f"  Cycles: {piddrl_metrics['cycles']:.0f}")
    print(f"  Avg Cycle Duration: {piddrl_metrics.get('avg_cycle_duration', 0):.1f} min")
    print(f"  Short-Cycling Violations: {piddrl_metrics.get('short_cycling_violations', 0):.0f}")
    print()
    
    print("DRL (No Cycling Penalty):")
    print(f"  Cost: ${drl_no_penalty_metrics['cost']:.2f}")
    print(f"  Discomfort: {drl_no_penalty_metrics['comfort']:.2f} degree-hours")
    print(f"  Cycles: {drl_no_penalty_metrics['cycles']:.0f}")
    print(f"  Avg Cycle Duration: {drl_no_penalty_metrics.get('avg_cycle_duration', 0):.1f} min")
    print(f"  Short-Cycling Violations: {drl_no_penalty_metrics.get('short_cycling_violations', 0):.0f}")
    print()
    
    # Calculate improvements
    cost_improvement = ((baseline_metrics['cost'] - piddrl_metrics['cost']) / baseline_metrics['cost']) * 100
    discomfort_improvement = ((baseline_metrics['comfort'] - piddrl_metrics['comfort']) / baseline_metrics['comfort']) * 100
    cycle_improvement = ((baseline_metrics['cycles'] - piddrl_metrics['cycles']) / baseline_metrics['cycles']) * 100
    
    print("Improvements (PI-DRL vs Baseline):")
    print(f"  Cost Reduction: {cost_improvement:.1f}%")
    print(f"  Discomfort Reduction: {discomfort_improvement:.1f}%")
    print(f"  Cycle Reduction: {cycle_improvement:.1f}%")
    print()
    
    # Generate tables
    print("Step 3: Generating tables...")
    table_gen = TableGenerator()
    df1, df2, df3 = table_gen.generate_all_tables(
        env=piddrl_env,
        baseline_metrics=baseline_metrics,
        piddrl_metrics=piddrl_metrics,
        no_cycling_penalty_metrics=drl_no_penalty_metrics,
        save_dir="./tables_real_data"
    )
    print("✓ Tables generated")
    print()
    
    return {
        'baseline': baseline_metrics,
        'piddrl': piddrl_metrics,
        'drl_no_penalty': drl_no_penalty_metrics,
        'data_source': 'real' if data_path else 'synthetic'
    }


def run_comprehensive_test():
    """
    Run comprehensive test with real data and sensitivity analysis
    """
    print("=" * 80)
    print("COMPREHENSIVE TEST: REAL DATA + SENSITIVITY ANALYSIS")
    print("=" * 80)
    print()
    
    # Part 1: Test with real data (if available)
    print("PART 1: Testing with Real AMPds2 Data")
    print("-" * 80)
    results = test_with_real_data()
    print()
    
    # Part 2: Sensitivity Analysis
    print("PART 2: Sensitivity Analysis")
    print("-" * 80)
    analyzer = SensitivityAnalyzer()
    sensitivity_results = analyzer.comprehensive_analysis(
        save_dir="./sensitivity_results",
        n_samples=2000
    )
    print()
    
    # Summary
    print("=" * 80)
    print("COMPREHENSIVE TEST COMPLETED")
    print("=" * 80)
    print()
    print("Results saved to:")
    print("  - Tables: ./tables_real_data/")
    print("  - Sensitivity Analysis: ./sensitivity_results/")
    print()
    
    return results, sensitivity_results


if __name__ == "__main__":
    # Check for real data first
    download_ampds2_sample()
    
    # Run comprehensive test
    results, sensitivity_results = run_comprehensive_test()
