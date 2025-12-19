"""
01_initial_exploration.py

Quick diagnostic check of NHL shot data.
Purpose: Verify data quality and understand basic structure before deep analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
from pathlib import Path

# --- set up paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "figures"

# Make sure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*70)
print("NHL SHOT DATA - INITIAL EXPLORATION")
print("="*70)
print(f"\nProject root: {PROJECT_ROOT}")
print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# --- load data ---
print("="*70)
print("LOADING DATA")
print("="*70)

data_file = DATA_DIR / "shots_2024.csv"
df = pd.read_csv(data_file)
print(f"Data loaded from {data_file}")
print(f"   Rows: {len(df):,}")
print(f"   Columns: {len(df.columns)}")

# --- basic overview ---
print("="*70)
print("DATASET OVERVIEW")
print("="*70)

print(f"Data shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

print("\nFirst 10 columns:")
for i, col in enumerate(df.columns[:10], 1):
    print(f" {i:2d}. {col}")
print(f"... and {len(df.columns) - 10} more columns.")

# --- target variable: goals ---
print("="*70)
print("TARGET VARIABLE: GOALS")
print("="*70)

if 'goal' in df.columns:
    n_goals = df['goal'].sum()
    n_shots = len(df)
    goal_rate = df['goal'].mean()

    print(f"\nGoal Statistics:")
    print(f"   Total shots: {n_shots:,}")
    print(f"   Total goals: {n_goals:,}")
    print(f"   Goal rate: {goal_rate:.2%}")
    print(f"   Non-goals: {n_shots - n_goals:,}")

    if goal_rate < 0.03:
        print("   Very low goal rate - may need oversampling")
    elif goal_rate > 0.15:
        print("   Unusually high goal rate - check data filtering")
    else:
        print("   Goal rate looks reasonable for NHL data")
else:
    print("ERROR: 'goal' column not found!")
    print("Available columns:", df.columns.tolist())
    sys.exit(1)

# --- key predictor variables ---
print("="*70)
print("KEY PREDICTOR VARIABLES")
print("="*70)

# distance
if 'shotDistance' in df.columns:
    print(f"\nShot Distance (feet):")
    print(f"   Mean: {df['shotDistance'].mean():.1f}")
    print(f"   Median: {df['shotDistance'].median():.1f}")
    print(f"   Std Dev: {df['shotDistance'].std():.1f}")
    print(f"   Range: [{df['shotDistance'].min():.1f}, {df['shotDistance'].max():.1f}]")
    print(f"   Missing: {df['shotDistance'].isnull().sum()} ({df['shotDistance'].isnull().mean():.1%})")
    
    # Check for issues
    if (df['shotDistance'] < 0).any():
        print(f"   ISSUE: {(df['shotDistance'] < 0).sum()} negative distances found")
    if (df['shotDistance'] > 200).any():
        print(f"   WARNING: {(df['shotDistance'] > 200).sum()} shots beyond 200 feet")
else:
    print("'shotDistance' column not found")

# angle
if 'shotAngle' in df.columns:
    print(f"\nShot Angle (degrees):")
    print(f"   Mean: {df['shotAngle'].mean():.1f}")
    print(f"   Median: {df['shotAngle'].median():.1f}")
    print(f"   Std Dev: {df['shotAngle'].std():.1f}")
    print(f"   Range: [{df['shotAngle'].min():.1f}, {df['shotAngle'].max():.1f}]")
    print(f"   Missing: {df['shotAngle'].isnull().sum()} ({df['shotAngle'].isnull().mean():.1%})")
else:
    print("'shotAngle' column not found")

# rebounds
print(f"\nRebound Identification:")
if 'shotRebound' in df.columns:
    print("   'shotRebound' column found")
    df['is_rebound'] = df['shotRebound'].astype(int)
    n_rebounds = df['is_rebound'].sum()
    print(f"   Rebounds: {n_rebounds:,} ({n_rebounds/len(df):.1%})")
    
    # Compare goal rates
    reb_goal_rate = df[df['is_rebound']==1]['goal'].mean()
    non_reb_goal_rate = df[df['is_rebound']==0]['goal'].mean()
    print(f"   Goal rate (rebounds): {reb_goal_rate:.2%}")
    print(f"   Goal rate (non-rebounds): {non_reb_goal_rate:.2%}")
    
    if reb_goal_rate > 0 and non_reb_goal_rate > 0:
        print(f"   Rebound advantage: {reb_goal_rate/non_reb_goal_rate:.2f}x")
    
elif 'lastEventCategory' in df.columns:
    print("   Using 'lastEventCategory' to identify rebounds")
    df['is_rebound'] = (df['lastEventCategory'].str.upper() == 'SHOT').astype(int)
    n_rebounds = df['is_rebound'].sum()
    print(f"   Rebounds: {n_rebounds:,} ({n_rebounds/len(df):.1%})")
else:
    print("   No obvious rebound indicator found")
    print("   Available columns with 'shot' or 'rebound':")
    rebound_cols = [c for c in df.columns if 'shot' in c.lower() or 'rebound' in c.lower()]
    for col in rebound_cols[:10]:
        print(f"      - {col}")

# --- quick visualizations ---
print("="*70)
print("CREATING QUICK VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Initial Data Exploration - NHL Shots 2024", fontsize=16, fontweight='bold')

# 1. Distance distribution
if 'shotDistance' in df.columns:
    axes[0, 0].hist(df['shotDistance'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(df['shotDistance'].mean(), color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {df["shotDistance"].mean():.1f} ft')
    axes[0, 0].set_xlabel('Shot Distance (feet)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('(A) Shot Distance Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)

# 2. Angle distribution
if 'shotAngle' in df.columns:
    axes[0, 1].hist(df['shotAngle'], bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[0, 1].axvline(df['shotAngle'].mean(), color='red', linestyle='--', linewidth=2,
                        label=f'Mean: {df["shotAngle"].mean():.1f}°')
    axes[0, 1].set_xlabel('Shot Angle (degrees)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('(B) Shot Angle Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)

# 3. Distance vs goal rate
if 'shotDistance' in df.columns:
    # Bin distance and calculate goal rate
    dist_bins = pd.cut(df['shotDistance'], bins=20)
    goal_by_dist = df.groupby(dist_bins)['goal'].mean()
    bin_centers = [interval.mid for interval in goal_by_dist.index]
    
    axes[1, 0].plot(bin_centers, goal_by_dist, 'o-', linewidth=2, markersize=6, color='steelblue')
    axes[1, 0].set_xlabel('Shot Distance (feet)')
    axes[1, 0].set_ylabel('Goal Rate')
    axes[1, 0].set_title('(C) Goal Rate vs Distance')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, goal_by_dist.max() * 1.2)

# 4. Angle vs goal rate
if 'shotAngle' in df.columns:
    # Bin angle and calculate goal rate
    angle_bins = pd.cut(df['shotAngle'], bins=20)
    goal_by_angle = df.groupby(angle_bins)['goal'].mean()
    bin_centers = [interval.mid for interval in goal_by_angle.index]
    
    axes[1, 1].plot(bin_centers, goal_by_angle, 'o-', linewidth=2, markersize=6, color='coral')
    axes[1, 1].set_xlabel('Shot Angle (degrees)')
    axes[1, 1].set_ylabel('Goal Rate')
    axes[1, 1].set_title('(D) Goal Rate vs Angle')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, goal_by_angle.max() * 1.2)

plt.tight_layout()
output_file = OUTPUT_DIR / "initial_exploration.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Figure saved: {output_file}")
plt.close()

# data quality summary
print("\n" + "="*70)
print("DATA QUALITY SUMMARY")
print("="*70)

issues = []
warnings = []

# --- check for critical issues ---
if 'shotDistance' in df.columns:
    if df['shotDistance'].isnull().sum() > len(df) * 0.05:
        issues.append(f"Distance missing in {df['shotDistance'].isnull().sum()} rows (>{5}%)")
    if (df['shotDistance'] < 0).any():
        issues.append(f"Negative distances: {(df['shotDistance'] < 0).sum()} rows")

if 'shotAngle' in df.columns:
    if df['shotAngle'].isnull().sum() > len(df) * 0.05:
        issues.append(f"Angle missing in {df['shotAngle'].isnull().sum()} rows (>{5}%)")

# check for warnings
if 'shotDistance' in df.columns:
    if (df['shotDistance'] > 200).any():
        warnings.append(f"Very long shots (>200 ft): {(df['shotDistance'] > 200).sum()} rows")

if issues:
    print("\nCRITICAL ISSUES (must fix):")
    for issue in issues:
        print(f"{issue}")
else:
    print("\nNo critical data quality issues detected")

if warnings:
    print("\nWARNINGS (should investigate):")
    for warning in warnings:
        print(f"{warning}")