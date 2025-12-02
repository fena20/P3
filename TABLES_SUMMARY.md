# Publication-Quality Tables Summary

## Overview

Three critical tables have been generated for your Applied Energy (Q1 Journal) submission. These tables provide the "precise, citable data" that reviewers require to validate your claims.

## Table 1: Simulation Parameters and Hyperparameters

**Purpose**: Strict Reproducibility

**Location**: `./tables/table1_hyperparameters.tex` and `.csv`

**Content**:
- **Building Physics Parameters**: R, C, HVAC Power, Time Step, Setpoints, Minimum Cycle Time
- **PPO Algorithm Hyperparameters**: Learning Rate, Discount Factor, GAE Lambda, Clip Range, Batch Size, etc.
- **Reward Function Weights**: w₁ (Cost), w₂ (Discomfort), w₃ (Cycling Penalty)

**Why It's Critical**: Reviewers need exact parameters to reproduce your results. Without this, results are considered "black-box" and untrustworthy.

## Table 2: Quantitative Performance Comparison

**Purpose**: Complement Radar Chart with Hard Numbers

**Location**: `./tables/table2_performance.tex` and `.csv`

**Content**:
- **Metrics**: Total Energy Cost, Thermal Discomfort, Equipment Switching Cycles, Peak Load, Carbon Emissions
- **Comparison**: Baseline vs. PI-DRL with exact values
- **Improvements**: Percentage reductions and absolute changes

**Why It's Critical**: Figures show trends, but tables provide the precise numbers reviewers cite. This quantifies exactly how much money/energy was saved.

## Table 3: Ablation Study (Physics-Informed Validation)

**Purpose**: Prove the Value of "Physics-Informed" Aspect

**Location**: `./tables/table3_ablation.tex` and `.csv`

**Content**:
- **Three Methods**: Baseline Thermostat, DRL (No Cycling Penalty), PI-DRL (With Cycling Penalty)
- **Key Metrics**: Cost, Discomfort, Cycles, Average Cycle Duration, Short-Cycling Violations
- **Critical Insight**: Shows that without cycling penalty, DRL might save money but destroys hardware through short-cycling

**Why It's Critical**: This answers the reviewer's question: "What happens if you remove the Cycling Penalty?" It demonstrates that the physics-informed constraint is essential for hardware protection.

## Key Findings from Ablation Study

The ablation study reveals:

1. **DRL without Cycling Penalty**:
   - May achieve lower energy costs
   - BUT: Causes excessive short-cycling violations
   - Result: Hardware degradation and potential equipment failure

2. **PI-DRL with Cycling Penalty**:
   - Maintains energy efficiency
   - Prevents short-cycling violations
   - Protects hardware while optimizing performance

## File Formats

Each table is available in two formats:

1. **LaTeX (.tex)**: Ready for direct inclusion in LaTeX manuscript
   - Properly formatted with `\begin{table}`, `\caption`, `\label`
   - Uses `booktabs` package formatting (`\toprule`, `\midrule`, `\bottomrule`)

2. **CSV (.csv)**: For Excel, data analysis, or conversion to other formats
   - Easy to import into spreadsheet software
   - Useful for further analysis or custom formatting

## Usage in Manuscript

### LaTeX Integration

Simply include the `.tex` files in your manuscript:

```latex
\input{tables/table1_hyperparameters.tex}
\input{tables/table2_performance.tex}
\input{tables/table3_ablation.tex}
```

### Manual Integration

If you prefer to copy-paste, the LaTeX tables are ready to use. Just ensure you have the `booktabs` package:

```latex
\usepackage{booktabs}
```

## Table Locations

All tables are saved in `./tables/`:

```
tables/
├── table1_hyperparameters.tex    # LaTeX format
├── table1_hyperparameters.csv    # CSV format
├── table2_performance.tex        # LaTeX format
├── table2_performance.csv        # CSV format
├── table3_ablation.tex            # LaTeX format
└── table3_ablation.csv           # CSV format
```

## Regenerating Tables

To regenerate tables with updated data:

```bash
python3 generate_tables_demo.py
```

Or generate everything (figures + tables):

```bash
python3 generate_all_outputs.py
```

## Next Steps

1. **Review Tables**: Check that all values match your simulation results
2. **Customize**: Modify table formatting if needed for journal requirements
3. **Integrate**: Add tables to your manuscript
4. **Validate**: Ensure all numbers are consistent with figures and text

## Notes for Reviewers

These tables address common reviewer concerns:

- **Reproducibility**: Table 1 provides all necessary parameters
- **Quantification**: Table 2 gives exact performance numbers
- **Validation**: Table 3 proves the physics-informed approach is essential

All tables follow Applied Energy formatting standards and are ready for submission.
