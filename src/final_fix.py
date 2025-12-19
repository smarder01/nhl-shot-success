"""
Final fix for coefficient signs:
1. Verify rebound coding with original data
2. Orthogonalize distance squared term
3. Check for any data errors
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

print("="*70)
print("FINAL DIAGNOSTIC AND FIX")
print("="*70)

# Load original data
print("\nLoading original data...")
df_orig = pd.read_csv(DATA_RAW / "shots_2024.csv")

# Check rebound column
print("\nChecking rebound column in original data...")
print(f"\nColumn 'shotRebound' unique values:")
print(df_orig['shotRebound'].value_counts())

print(f"\nGoal rate by shotRebound value:")
rebound_check = df_orig.groupby('shotRebound')['goal'].agg(['count', 'sum', 'mean'])
print(rebound_check)

# Determine correct coding
if rebound_check.loc[1, 'mean'] > rebound_check.loc[0, 'mean']:
    print("\nCoding is correct: shotRebound=1 means rebound, and they score more")
    rebound_correct = True
else:
    print("\nCoding might be FLIPPED!")
    rebound_correct = False

# ============================================================================
# RELOAD AND REPROCESS
# ============================================================================
print("\nReprocessing data...")

df = df_orig.copy()

# Clean
df = df.dropna(subset=['shotDistance', 'shotAngle', 'goal'])
df = df[(df['shotDistance'] >= 1) & (df['shotDistance'] <= 100)]
df = df[df['shotAngle'].abs() <= 89]
df = df[df['goal'].isin([0, 1])]

# Create features
df['angle_abs'] = df['shotAngle'].abs()

# CRITICAL: Ensure rebound is coded correctly
if 'shotRebound' in df.columns:
    if rebound_correct:
        df['is_rebound'] = df['shotRebound'].astype(int)
    else:
        print("FLIPPING rebound coding!")
        df['is_rebound'] = (1 - df['shotRebound']).astype(int)
else:
    df['is_rebound'] = 0

# Verify after coding
print(f"\n   Verification after coding:")
print(df.groupby('is_rebound')['goal'].agg(['count', 'mean']))

# Standardize
distance_mean = df['shotDistance'].mean()
distance_std_val = df['shotDistance'].std()
df['distance_std'] = (df['shotDistance'] - distance_mean) / distance_std_val

angle_mean = df['angle_abs'].mean()
angle_std_val = df['angle_abs'].std()
df['angle_abs_std'] = (df['angle_abs'] - angle_mean) / angle_std_val

# ORTHOGONALIZE distance squared
df['distance_std_sq_raw'] = df['distance_std'] ** 2
dist_sq_mean = df['distance_std_sq_raw'].mean()
df['distance_std_sq'] = df['distance_std_sq_raw'] - dist_sq_mean

print(f"\n   Distance squared orthogonalization:")
print(f"   Raw mean: {dist_sq_mean:.4f}")
print(f"   Centered mean: {df['distance_std_sq'].mean():.6f}")
print(f"   Correlation(dist, dist²): {df[['distance_std', 'distance_std_sq']].corr().iloc[0,1]:.4f}")

# ============================================================================
# SAVE FINAL VERSION
# ============================================================================
print("\nSaving final dataset...")

analysis_df = df[[
    'shotDistance',
    'shotAngle',
    'angle_abs',
    'distance_std',
    'angle_abs_std',
    'distance_std_sq',
    'is_rebound',
    'goal'
]].copy()

final_path = DATA_PROCESSED / "shots_analysis_ready_FINAL.csv"
analysis_df.to_csv(final_path, index=False)
print(f"   Saved: {final_path.name}")

# Save params
params = pd.DataFrame({
    'parameter': ['distance_mean', 'distance_std', 'angle_abs_mean', 'angle_abs_std', 
                  'distance_sq_mean'],
    'value': [distance_mean, distance_std_val, angle_mean, angle_std_val, dist_sq_mean]
})
params_path = DATA_PROCESSED / "standardization_parameters_FINAL.csv"
params.to_csv(params_path, index=False)
print(f"   Saved: {params_path.name}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nDataset: {len(analysis_df):,} rows")
print(f"\nGoal rate overall: {analysis_df['goal'].mean():.3%}")
print(f"Goal rate (non-rebounds): {analysis_df[analysis_df['is_rebound']==0]['goal'].mean():.3%}")
print(f"Goal rate (rebounds): {analysis_df[analysis_df['is_rebound']==1]['goal'].mean():.3%}")
print(f"Rebound advantage: {analysis_df[analysis_df['is_rebound']==1]['goal'].mean() / analysis_df[analysis_df['is_rebound']==0]['goal'].mean():.2f}x")
