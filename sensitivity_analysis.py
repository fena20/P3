"""
Sensitivity Analysis Module for Physics-Informed DRL Framework
Analyzes sensitivity to key parameters: R, C, reward weights, cycle time, etc.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from smarthome_env import SmartHomeEnv
from generate_tables_demo import (
    run_baseline_thermostat, run_piddrl_with_cycling_penalty, run_drl_no_cycling_penalty
)
import warnings
warnings.filterwarnings('ignore')


class SensitivityAnalyzer:
    """
    Performs comprehensive sensitivity analysis on key parameters
    """
    
    def __init__(self, base_params: Dict = None):
        """
        Initialize sensitivity analyzer
        
        Parameters:
        -----------
        base_params : dict
            Base parameter values
        """
        if base_params is None:
            # Default base parameters
            self.base_params = {
                'R': 0.05,  # Thermal resistance
                'C': 0.5,   # Thermal capacitance
                'w1': 1.0,  # Cost weight
                'w2': 10.0, # Discomfort weight
                'w3': 5.0,  # Cycling penalty weight
                'min_cycle_time': 15,  # Minimum cycle time (minutes)
                'Q_hvac_max': 3.0,  # Maximum HVAC power
                'T_setpoint': 22.0,  # Comfort setpoint
                'T_tolerance': 1.5   # Comfort tolerance
            }
        else:
            self.base_params = base_params
    
    def analyze_parameter_sensitivity(self, param_name: str, param_values: List[float],
                                    n_samples: int = 2000) -> pd.DataFrame:
        """
        Analyze sensitivity to a single parameter
        
        Parameters:
        -----------
        param_name : str
            Name of parameter to vary
        param_values : list
            List of parameter values to test
        n_samples : int
            Number of simulation steps
        
        Returns:
        --------
        pd.DataFrame
            Results with metrics for each parameter value
        """
        results = []
        
        print(f"Analyzing sensitivity to {param_name}...")
        print(f"  Testing {len(param_values)} values: {param_values}")
        
        for val in param_values:
            # Create environment with modified parameter
            params = self.base_params.copy()
            params[param_name] = val
            
            try:
                env = SmartHomeEnv(
                    R=params['R'],
                    C=params['C'],
                    w1=params['w1'],
                    w2=params['w2'],
                    w3=params['w3'],
                    min_cycle_time=params['min_cycle_time'],
                    Q_hvac_max=params['Q_hvac_max'],
                    T_setpoint=params['T_setpoint'],
                    T_tolerance=params['T_tolerance']
                )
                
                # Run PI-DRL simulation
                metrics = run_piddrl_with_cycling_penalty(env, n_samples=n_samples)
                
                results.append({
                    'parameter': param_name,
                    'value': val,
                    'cost': metrics['cost'],
                    'discomfort': metrics['comfort'],
                    'cycles': metrics['cycles'],
                    'avg_cycle_duration': metrics.get('avg_cycle_duration', 0),
                    'short_cycling_violations': metrics.get('short_cycling_violations', 0),
                    'peak_load': metrics['peak_load']
                })
                
            except Exception as e:
                print(f"    Error at {param_name}={val}: {e}")
                continue
        
        df = pd.DataFrame(results)
        return df
    
    def analyze_thermal_resistance(self, R_values: List[float] = None,
                                  n_samples: int = 2000) -> pd.DataFrame:
        """
        Analyze sensitivity to thermal resistance R
        
        Parameters:
        -----------
        R_values : list
            List of R values to test (default: [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
        n_samples : int
            Number of simulation steps
        
        Returns:
        --------
        pd.DataFrame
            Sensitivity analysis results
        """
        if R_values is None:
            R_values = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
        
        return self.analyze_parameter_sensitivity('R', R_values, n_samples)
    
    def analyze_thermal_capacitance(self, C_values: List[float] = None,
                                   n_samples: int = 2000) -> pd.DataFrame:
        """
        Analyze sensitivity to thermal capacitance C
        
        Parameters:
        -----------
        C_values : list
            List of C values to test (default: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        n_samples : int
            Number of simulation steps
        
        Returns:
        --------
        pd.DataFrame
            Sensitivity analysis results
        """
        if C_values is None:
            C_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        return self.analyze_parameter_sensitivity('C', C_values, n_samples)
    
    def analyze_reward_weights(self, w1_values: List[float] = None,
                              w2_values: List[float] = None,
                              w3_values: List[float] = None,
                              n_samples: int = 2000) -> Dict[str, pd.DataFrame]:
        """
        Analyze sensitivity to reward function weights
        
        Parameters:
        -----------
        w1_values : list
            Cost weight values (default: [0.5, 1.0, 1.5, 2.0])
        w2_values : list
            Discomfort weight values (default: [5.0, 10.0, 15.0, 20.0])
        w3_values : list
            Cycling penalty weight values (default: [0, 2.5, 5.0, 7.5, 10.0])
        n_samples : int
            Number of simulation steps
        
        Returns:
        --------
        dict
            Dictionary with results for w1, w2, w3
        """
        if w1_values is None:
            w1_values = [0.5, 1.0, 1.5, 2.0]
        if w2_values is None:
            w2_values = [5.0, 10.0, 15.0, 20.0]
        if w3_values is None:
            w3_values = [0, 2.5, 5.0, 7.5, 10.0]
        
        results = {}
        
        print("Analyzing sensitivity to reward weights...")
        results['w1'] = self.analyze_parameter_sensitivity('w1', w1_values, n_samples)
        results['w2'] = self.analyze_parameter_sensitivity('w2', w2_values, n_samples)
        results['w3'] = self.analyze_parameter_sensitivity('w3', w3_values, n_samples)
        
        return results
    
    def analyze_cycle_time(self, cycle_times: List[int] = None,
                         n_samples: int = 2000) -> pd.DataFrame:
        """
        Analyze sensitivity to minimum cycle time
        
        Parameters:
        -----------
        cycle_times : list
            List of cycle times in minutes (default: [5, 10, 15, 20, 25, 30])
        n_samples : int
            Number of simulation steps
        
        Returns:
        --------
        pd.DataFrame
            Sensitivity analysis results
        """
        if cycle_times is None:
            cycle_times = [5, 10, 15, 20, 25, 30]
        
        return self.analyze_parameter_sensitivity('min_cycle_time', cycle_times, n_samples)
    
    def plot_sensitivity(self, df: pd.DataFrame, param_name: str,
                        metrics: List[str] = None, save_path: str = None):
        """
        Plot sensitivity analysis results
        
        Parameters:
        -----------
        df : pd.DataFrame
            Sensitivity analysis results
        param_name : str
            Name of parameter analyzed
        metrics : list
            List of metrics to plot (default: ['cost', 'discomfort', 'cycles'])
        save_path : str
            Path to save figure
        """
        if metrics is None:
            metrics = ['cost', 'discomfort', 'cycles']
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            ax.plot(df['value'], df[metric], 'o-', linewidth=2, markersize=8)
            ax.set_xlabel(f'{param_name}', fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Sensitivity: {metric.replace("_", " ").title()}', fontsize=13)
        
        plt.suptitle(f'Sensitivity Analysis: {param_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sensitivity plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def comprehensive_analysis(self, save_dir: str = "./sensitivity_results",
                               n_samples: int = 2000) -> Dict[str, pd.DataFrame]:
        """
        Perform comprehensive sensitivity analysis on all key parameters
        
        Parameters:
        -----------
        save_dir : str
            Directory to save results
        n_samples : int
            Number of simulation steps
        
        Returns:
        --------
        dict
            Dictionary with all analysis results
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("=" * 80)
        print("COMPREHENSIVE SENSITIVITY ANALYSIS")
        print("=" * 80)
        print()
        
        results = {}
        
        # 1. Thermal Resistance (R)
        print("1. Analyzing Thermal Resistance (R)...")
        results['R'] = self.analyze_thermal_resistance(n_samples=n_samples)
        results['R'].to_csv(f"{save_dir}/sensitivity_R.csv", index=False)
        self.plot_sensitivity(results['R'], 'R', 
                            metrics=['cost', 'discomfort', 'cycles'],
                            save_path=f"{save_dir}/sensitivity_R.png")
        print()
        
        # 2. Thermal Capacitance (C)
        print("2. Analyzing Thermal Capacitance (C)...")
        results['C'] = self.analyze_thermal_capacitance(n_samples=n_samples)
        results['C'].to_csv(f"{save_dir}/sensitivity_C.csv", index=False)
        self.plot_sensitivity(results['C'], 'C',
                            metrics=['cost', 'discomfort', 'cycles'],
                            save_path=f"{save_dir}/sensitivity_C.png")
        print()
        
        # 3. Reward Weights
        print("3. Analyzing Reward Weights (w1, w2, w3)...")
        weight_results = self.analyze_reward_weights(n_samples=n_samples)
        results.update(weight_results)
        for w_name, w_df in weight_results.items():
            w_df.to_csv(f"{save_dir}/sensitivity_{w_name}.csv", index=False)
            self.plot_sensitivity(w_df, w_name,
                                metrics=['cost', 'discomfort', 'cycles'],
                                save_path=f"{save_dir}/sensitivity_{w_name}.png")
        print()
        
        # 4. Minimum Cycle Time
        print("4. Analyzing Minimum Cycle Time...")
        results['cycle_time'] = self.analyze_cycle_time(n_samples=n_samples)
        results['cycle_time'].to_csv(f"{save_dir}/sensitivity_cycle_time.csv", index=False)
        self.plot_sensitivity(results['cycle_time'], 'min_cycle_time',
                            metrics=['cost', 'cycles', 'short_cycling_violations'],
                            save_path=f"{save_dir}/sensitivity_cycle_time.png")
        print()
        
        # Generate summary report
        self._generate_summary_report(results, save_dir)
        
        print("=" * 80)
        print("Sensitivity analysis completed!")
        print(f"Results saved to: {save_dir}/")
        print("=" * 80)
        
        return results
    
    def _generate_summary_report(self, results: Dict[str, pd.DataFrame], save_dir: str):
        """
        Generate summary report of sensitivity analysis
        """
        report_path = f"{save_dir}/sensitivity_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SENSITIVITY ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            for param_name, df in results.items():
                f.write(f"\n{param_name.upper()} Sensitivity:\n")
                f.write("-" * 80 + "\n")
                
                # Calculate sensitivity coefficients
                base_value = self.base_params.get(param_name, df['value'].iloc[len(df)//2])
                base_idx = df['value'].sub(base_value).abs().idxmin()
                base_metrics = df.iloc[base_idx]
                
                f.write(f"Base value: {base_value}\n")
                f.write(f"Base metrics:\n")
                f.write(f"  Cost: ${base_metrics['cost']:.2f}\n")
                f.write(f"  Discomfort: {base_metrics['discomfort']:.2f}\n")
                f.write(f"  Cycles: {base_metrics['cycles']:.0f}\n\n")
                
                # Variation analysis
                f.write("Variation Analysis:\n")
                for metric in ['cost', 'discomfort', 'cycles']:
                    if metric in df.columns:
                        min_val = df[metric].min()
                        max_val = df[metric].max()
                        range_val = max_val - min_val
                        base_val = base_metrics[metric]
                        variation_pct = (range_val / base_val * 100) if base_val > 0 else 0
                        
                        f.write(f"  {metric}: Range=[{min_val:.2f}, {max_val:.2f}], "
                               f"Variation={variation_pct:.1f}%\n")
                
                f.write("\n")
        
        print(f"Summary report saved to {report_path}")
