"""
fixed version addressing coefficient sign issues:
- Use ABSOLUTE angle (removes directional ambiguity)
- Check for multicollinearity
- Verify rebound coding
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

DATA_PROCESSED.mkdir(exist_ok=True)

print("="*70)
print("FIXED DATA PREPROCESSING")
print("="*70)

# ============================================================================
# LOAD RAW DATA
# ============================================================================
print("\nLoading raw data...")
df = pd.read_csv(DATA_RAW / "shots_2024.csv")
print(f"   Initial dataset: {len(df):,} rows")

# Create rebound indicator
if 'shotRebound' in df.columns:
    df['is_rebound'] = df['shotRebound'].astype(int)
else:
    print("   Warning: No rebound column found")
    df['is_rebound'] = 0

# ============================================================================
# DATA CLEANING
# ============================================================================
print("\nCleaning data...")

initial_count = len(df)
key_columns = ['shotDistance', 'shotAngle', 'goal']

# Remove missing values
df = df.dropna(subset=key_columns)

# Remove unrealistic values
df = df[(df['shotDistance'] >= 1) & (df['shotDistance'] <= 100)]
df = df[df['shotAngle'].abs() <= 89]
df = df[df['goal'].isin([0, 1])]

print(f"   Clean dataset: {len(df):,} rows ({len(df)/initial_count:.1%} retained)")

# ============================================================================
# FIX: USE ABSOLUTE ANGLE
# ============================================================================
print("\nCreating features (FIXED)...")

# Use ABSOLUTE angle (distance from center, ignoring left/right)
df['angle_abs'] = df['shotAngle'].abs()

print(f"\n   IMPORTANT CHANGE: Using absolute angle")
print(f"   Original angle range: [{df['shotAngle'].min():.1f}°, {df['shotAngle'].max():.1f}°]")
print(f"   Absolute angle range: [{df['angle_abs'].min():.1f}°, {df['angle_abs'].max():.1f}°]")

# Verify rebound coding
print(f"\n   Rebound verification:")
reb_stats = df.groupby('is_rebound')['goal'].agg(['mean', 'count', 'sum'])
print(reb_stats)
if df[df['is_rebound']==1]['goal'].mean() > df[df['is_rebound']==0]['goal'].mean():
    print(f"   Rebound coding correct: {df[df['is_rebound']==1]['goal'].mean():.3f} > {df[df['is_rebound']==0]['goal'].mean():.3f}")
else:
    print(f"   WARNING: Rebounds score LESS? Something wrong!")

# ============================================================================
# STANDARDIZATION
# ============================================================================
print("\nStandardizing predictors...")

# Distance
distance_mean = df['shotDistance'].mean()
distance_std = df['shotDistance'].std()
df['distance_std'] = (df['shotDistance'] - distance_mean) / distance_std

# ABSOLUTE Angle (key change!)
angle_mean = df['angle_abs'].mean()
angle_std = df['angle_abs'].std()
df['angle_abs_std'] = (df['angle_abs'] - angle_mean) / angle_std

# Squared distance
df['distance_std_sq'] = df['distance_std'] ** 2

print(f"\n   Distance standardization:")
print(f"   Mean: {distance_mean:.2f} ft, Std: {distance_std:.2f} ft")
print(f"   Standardized: mean={df['distance_std'].mean():.6f}, std={df['distance_std'].std():.6f}")

print(f"\n   Absolute angle standardization:")
print(f"   Mean: {angle_mean:.2f}°, Std: {angle_std:.2f}°")
print(f"   Standardized: mean={df['angle_abs_std'].mean():.6f}, std={df['angle_abs_std'].std():.6f}")

# ============================================================================
# CHECK MULTICOLLINEARITY
# ============================================================================
print("\nChecking multicollinearity...")

X_check = df[['distance_std', 'distance_std_sq', 'angle_abs_std', 'is_rebound']]
corr_matrix = X_check.corr()

print(f"\n   Correlation matrix:")
print(corr_matrix.round(3))

print(f"\n   Key correlations:")
print(f"   • distance_std × distance_std_sq: {corr_matrix.loc['distance_std', 'distance_std_sq']:.3f}")
if abs(corr_matrix.loc['distance_std', 'distance_std_sq']) > 0.8:
    print(f"     High correlation detected! May cause numerical issues.")
else:
    print(f"     Acceptable level")

# ============================================================================
# CREATE FINAL DATASET
# ============================================================================
print("\n" + "="*70)
print("Creating final analysis dataset...")

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

print(f"\n   Final dataset: {len(analysis_df):,} rows × {len(analysis_df.columns)} columns")

# Save
final_path = DATA_PROCESSED / "shots_analysis_ready_FIXED.csv"
analysis_df.to_csv(final_path, index=False)
print(f"   Saved: {final_path.name}")

# Save standardization params
params = pd.DataFrame({
    'parameter': ['distance_mean', 'distance_std', 'angle_abs_mean', 'angle_abs_std'],
    'value': [distance_mean, distance_std, angle_mean, angle_std]
})
params_path = DATA_PROCESSED / "standardization_parameters_FIXED.csv"
params.to_csv(params_path, index=False)
print(f"   Saved: {params_path.name}")

# ============================================================================
# PREVIEW
# ============================================================================
print("\n" + "="*70)
print("PREVIEW OF FIXED DATA")
print("="*70)
print("\nFirst 10 rows:")
print(analysis_df.head(10).to_string(index=False))

print("\n" + "="*70)
print("FIXED PREPROCESSING COMPLETE!")
print("="*70)
print("\nNext: Re-run baseline model with FIXED data")
print("  python src/04_baseline_model.py")
print("  (Update script to use 'shots_analysis_ready_FIXED.csv')")
print("="*70)