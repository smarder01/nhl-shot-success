"""
Clean data and prepare final analysis-ready dataset.
- Remove outliers
- Handle missing values
- Standardize continuous predictors
-Create final dataset for modeling
"""

import pandas as pd
import numpy as np
from pathlib import Path

# set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# create processed directory if it doesn't exist
DATA_PROCESSED.mkdir(exist_ok=True)

print("="*70)
print("DATA CLEANING AND PREPROCESSING")
print("="*70)

# --- load raw data ---
print("Loading raw data...")
df = pd.read_csv(DATA_RAW / "shots_2024.csv")
print(f"Initial dataset: {len(df):,} rows")

# create rebound indicator
if 'shotRebound' in df.columns:
    df['is_rebound'] = df['shotRebound'].astype(int)
else:
    print("Warning: No rebound column found")
    df['is_rebound'] = 0

# --- data cleaning ---
print("Cleaning data...")

initial_count = len(df)

# remove rows with missing values
key_columns = ['shotDistance', 'shotAngle', 'goal']
missing_before = df[key_columns].isnull().sum().sum()
df = df.dropna(subset=key_columns)
missing_removed = initial_count - len(df)
if missing_removed > 0:
    print(f"Removed {missing_removed} rows with missing values")

# remove unrealistic distances (keep shoes between 0 and 100 feet)
invalid_distance = ((df['shotDistance'] < 1) | (df['shotDistance'] > 100)).sum()
df = df[(df['shotDistance'] >= 1) & (df['shotDistance'] <= 100)]
if invalid_distance > 0:
    print(f"   Removed {invalid_distance} rows with invalid distance (<1 or >100 ft)")

# Remove extreme angles (beyond +/- 89 degrees - basically behind the net)
invalid_angle = (df['shotAngle'].abs() > 89).sum()
df = df[df['shotAngle'].abs() <= 89]
if invalid_angle > 0:
    print(f"   Removed {invalid_angle} rows with extreme angles (>89°)")

# Verify goal column is binary
if not df['goal'].isin([0, 1]).all():
    print("Warning: goal column contains non-binary values")
    df = df[df['goal'].isin([0, 1])]

print(f"Clean dataset: {len(df):,} rows ({len(df)/initial_count:.1%} retained)")

# Save cleaned data
cleaned_path = DATA_PROCESSED / "shots_cleaned.csv"
df[['shotDistance', 'shotAngle', 'is_rebound', 'goal']].to_csv(cleaned_path, index=False)
print(f"   Saved: {cleaned_path.name}")

# --- data quality reports---
print("Data quality report...")

print(f"\nFinal dataset statistics:")
print(f"   Total shots: {len(df):,}")
print(f"   Goals: {df['goal'].sum():,} ({df['goal'].mean():.2%})")
print(f"   Rebounds: {df['is_rebound'].sum():,} ({df['is_rebound'].mean():.1%})")
print(f"   Distance range: [{df['shotDistance'].min():.1f}, {df['shotDistance'].max():.1f}] ft")
print(f"   Angle range: [{df['shotAngle'].min():.1f}, {df['shotAngle'].max():.1f}]°")
print(f"   Missing values: {df[key_columns + ['is_rebound']].isnull().sum().sum()}")

# --- standardization ---
print("Standardizing continuous predictors...")

# calculate standatdization parameters
distance_mean = df['shotDistance'].mean()
distance_std = df['shotDistance'].std()
angle_mean = df['shotAngle'].mean()
angle_std = df['shotAngle'].std()

print(f"\n   Distance standardization:")
print(f"   Mean: {distance_mean:.2f} ft")
print(f"   Std Dev: {distance_std:.2f} ft")
print(f"   Formula: (distance - {distance_mean:.2f}) / {distance_std:.2f}")

print(f"\n   Angle standardization:")
print(f"   Mean: {angle_mean:.2f}°")
print(f"   Std Dev: {angle_std:.2f}°")
print(f"   Formula: (angle - {angle_mean:.2f}) / {angle_std:.2f}")

# Create standardized variables
df['distance_std'] = (df['shotDistance'] - distance_mean) / distance_std
df['angle_std'] = (df['shotAngle'] - angle_mean) / angle_std

# Verify standardization
print(f"\n   Standardized distance: mean={df['distance_std'].mean():.6f}, std={df['distance_std'].std():.6f}")
print(f"   Standardized angle: mean={df['angle_std'].mean():.6f}, std={df['angle_std'].std():.6f}")

# --- create final analysis dataset ---
print("Creating final analysis dataset...")

analysis_df = df[[
    'shotDistance',
    'shotAngle',
    'distance_std',
    'angle_std',
    'is_rebound',
    'goal'
]].copy()

# add squared distnace term (for quadratic model)
analysis_df['distance_std_sq'] = analysis_df['distance_std'] ** 2

# summary of final datset
print(f"\nFinal analysis dataset:")
print(f"\nFinal analysis dataset:")
print(f"   Rows: {len(analysis_df):,}")
print(f"   Columns: {len(analysis_df.columns)}")
print(f"\nColumn names:")
for i, col in enumerate(analysis_df.columns, 1):
    print(f"      {i}. {col}")

# Save final dataset
final_path = DATA_PROCESSED / "shots_analysis_ready.csv"
analysis_df.to_csv(final_path, index=False)
print(f"\n   Saved: {final_path.name}")

# Save standardization parameters (for later use in predictions)
params = pd.DataFrame({
    'parameter': ['distance_mean', 'distance_std', 'angle_mean', 'angle_std'],
    'value': [distance_mean, distance_std, angle_mean, angle_std]
})
params_path = DATA_PROCESSED / "standardization_parameters.csv"
params.to_csv(params_path, index=False)
print(f"   Saved: {params_path.name}")

# --- summary ---
print("="*70)
print("PREPROCESSNG SUMMARY")
print("="*70)

print(f"""
DATA CLEANING:
   • Starting rows: {initial_count:,}
   • Final rows: {len(df):,}
   • Rows removed: {initial_count - len(df):,} ({(initial_count - len(df))/initial_count:.1%})

FINAL DATASET:
   • Observations: {len(analysis_df):,}
   • Features: {len(analysis_df.columns) - 1} (+ 1 target)
   • Goal rate: {analysis_df['goal'].mean():.2%}
   • Rebound rate: {analysis_df['is_rebound'].mean():.1%}

SAVED FILES:
   • data/processed/shots_cleaned.csv
   • data/processed/shots_analysis_ready.csv
   • data/processed/standardization_parameters.csv

READY FOR MODELING!
   
   Preview of analysis data:
""")

print(analysis_df.head(10).to_string(index=False))