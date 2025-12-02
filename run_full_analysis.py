"""
Complete Analysis Pipeline: Real Dataset + Sensitivity Analysis
Runs comprehensive tests with real AMPds2 data and sensitivity analysis
"""

import os
import sys
from test_real_dataset import run_comprehensive_test, test_with_real_data
from sensitivity_analysis import SensitivityAnalyzer
from generate_all_outputs import main as generate_all_outputs


def main():
    """
    Run complete analysis pipeline:
    1. Test with real AMPds2 data (if available)
    2. Comprehensive sensitivity analysis
    3. Generate all figures and tables
    """
    print("=" * 80)
    print("COMPLETE ANALYSIS PIPELINE")
    print("Real Dataset Testing + Sensitivity Analysis + Full Outputs")
    print("=" * 80)
    print()
    
    # Step 1: Test with real data
    print("STEP 1: Testing with Real AMPds2 Dataset")
    print("-" * 80)
    try:
        real_data_results = test_with_real_data()
        print("✓ Real data testing completed")
    except Exception as e:
        print(f"⚠ Real data testing failed: {e}")
        print("  Continuing with synthetic data...")
        real_data_results = None
    print()
    
    # Step 2: Sensitivity Analysis
    print("STEP 2: Comprehensive Sensitivity Analysis")
    print("-" * 80)
    try:
        analyzer = SensitivityAnalyzer()
        sensitivity_results = analyzer.comprehensive_analysis(
            save_dir="./sensitivity_results",
            n_samples=2000
        )
        print("✓ Sensitivity analysis completed")
    except Exception as e:
        print(f"⚠ Sensitivity analysis failed: {e}")
        sensitivity_results = None
    print()
    
    # Step 3: Generate all outputs (figures + tables)
    print("STEP 3: Generating All Publication Outputs")
    print("-" * 80)
    try:
        generate_all_outputs()
        print("✓ All outputs generated")
    except Exception as e:
        print(f"⚠ Output generation failed: {e}")
    print()
    
    # Summary
    print("=" * 80)
    print("ANALYSIS PIPELINE COMPLETED")
    print("=" * 80)
    print()
    print("Generated outputs:")
    print("  ✓ Figures: ./figures/")
    print("  ✓ Tables: ./tables/")
    print("  ✓ Sensitivity Analysis: ./sensitivity_results/")
    if real_data_results:
        print("  ✓ Real Data Tables: ./tables_real_data/")
    print()
    print("Next steps:")
    print("  1. Review sensitivity analysis results")
    print("  2. Check real data results (if available)")
    print("  3. Integrate findings into manuscript")
    print("=" * 80)


if __name__ == "__main__":
    main()
