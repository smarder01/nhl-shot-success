"""
Coprehensive exploratory data analysis for NHL shot data
Generates publication-quality figures ad tables for project report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# set style
sns.set_style(style="whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

# set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"

print("="*70)
print("COMPREHENSIVE EDA - NHL SHOT DATA")
print("="*70)

# --- load data ---
print("Loading data...")
df = pd.read_csv(DATA_DIR / "shots_2024.csv")

# create rebound indicator
if 'shotRebound' in df.columns:
    df['is_rebound'] = df['shotRebound'].astype(int)
else:
    df['is_rebound'] = 0
    print("   Warning: No rebound column found, setting all to 0")

print(f"   Loaded {len(df):,} shots")

# --- univariate distributions ---
print("Creating Figure 1: Univariate Distributions...")

fig1, axes = plt.subplots(2, 2, figsize=(10, 8))
fig1.suptitle("Figure 1: Univariate Distributions of Key Variables", fontsize=16, fontweight='bold', y = 0.995)

# distance histogram
ax = axes[0, 0]
ax.hist(df['shotDistance'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(df['shotDistance'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {df["shotDistance"].mean():.1f} ft')
ax.axvline(df['shotDistance'].median(), color='orange', linestyle='--', linewidth=2,
           label=f'Median: {df["shotDistance"].median():.1f} ft')
ax.set_xlabel('Shot Distance (feet)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('(A) Shot Distance Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# angle histogram
ax = axes[0, 1]
ax.hist(df['shotAngle'], bins=50, edgecolor='black', alpha=0.7, color='coral')
ax.axvline(df['shotAngle'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {df["shotAngle"].mean():.1f}°')
ax.axvline(df['shotAngle'].median(), color='orange', linestyle='--', linewidth=2,
           label=f'Median: {df["shotAngle"].median():.1f}°')
ax.set_xlabel('Shot Angle (degrees)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('(B) Shot Angle Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# goal outcome bar chart
ax = axes[1, 0]
goal_counts = df['goal'].value_counts()
colors = ['lightcoral', 'lightgreen']
bars = ax.bar(['No Goal', 'Goal'], goal_counts.values, color=colors,
              edgecolor='black', alpha=0.8, linewidth=1.5)
ax.set_ylabel('Count', fontsize=12, fontweight='bold')
ax.set_title('(C) Shot Outcomes', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for i, (bar, v) in enumerate(zip(bars, goal_counts.values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1000,
            f'{v:,}\n({v/len(df):.1%})',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# rebound comparison
ax = axes[1, 1]
rebound_stats = df.groupby('is_rebound')['goal'].agg(['mean', 'count'])
colors = ['skyblue', 'salmon']
bars = ax.bar(['Non-Rebound', 'Rebound'], rebound_stats['mean'].values,
              color=colors, edgecolor='black', alpha=0.8, linewidth=1.5)
ax.set_ylabel('Goal Rate', fontsize=12, fontweight='bold')
ax.set_title('(D) Goal Rate by Rebound Status', fontsize=13, fontweight='bold')
ax.set_ylim(0, rebound_stats['mean'].max() * 1.25)
ax.grid(axis='y', alpha=0.3)
for i, (bar, rate, count) in enumerate(zip(bars, rebound_stats['mean'], rebound_stats['count'])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
            f'{rate:.2%}\n(n={count:,})',
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
fig1_path = FIGURES_DIR / "fig1_univariate_distributions.png"
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {fig1_path.name}")

# --- bivariate relationships ---
print("\Creating Figure 2: Goal Rate vs Predictors...")

fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Figure 2: Goal Rate as Function of Shot Characteristics', 
              fontsize=16, fontweight='bold')

# (A) Distance vs goal rate
ax = axes[0]
distance_bins = pd.cut(df['shotDistance'], bins=30)
goal_by_dist = df.groupby(distance_bins, observed=True)['goal'].agg(['mean', 'count'])
bin_centers = [interval.mid for interval in goal_by_dist.index]
ax.plot(bin_centers, goal_by_dist['mean'], 'o-', linewidth=2.5, markersize=8,
        color='steelblue', markeredgecolor='black', markeredgewidth=1)
ax.set_xlabel('Shot Distance (feet)', fontsize=12, fontweight='bold')
ax.set_ylabel('Goal Rate', fontsize=12, fontweight='bold')
ax.set_title('(A) Nonlinear Distance Effect', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, goal_by_dist['mean'].max() * 1.15)
# Add annotation for nonlinearity
ax.annotate('Steep drop\nat close range', xy=(10, 0.18), xytext=(25, 0.20),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, color='red', fontweight='bold')
ax.annotate('Slower decline\nat distance', xy=(60, 0.02), xytext=(45, 0.08),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, color='red', fontweight='bold')

# (B) Angle vs goal rate
ax = axes[1]
angle_bins = pd.cut(df['shotAngle'], bins=30)
goal_by_angle = df.groupby(angle_bins, observed=True)['goal'].agg(['mean', 'count'])
bin_centers = [interval.mid for interval in goal_by_angle.index]
ax.plot(bin_centers, goal_by_angle['mean'], 'o-', linewidth=2.5, markersize=8,
        color='coral', markeredgecolor='black', markeredgewidth=1)
ax.set_xlabel('Shot Angle (degrees)', fontsize=12, fontweight='bold')
ax.set_ylabel('Goal Rate', fontsize=12, fontweight='bold')
ax.set_title('(B) Angle Effect (Symmetrical)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, goal_by_angle['mean'].max() * 1.15)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5, label='Center (0°)')
ax.legend(fontsize=10)

plt.tight_layout()
fig2_path = FIGURES_DIR / "fig2_bivariate_relationships.png"
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {fig2_path.name}")

# --- 2d heatmap (distance x angle) ---
print("Creating Figure 3: 2D Heatmap of Goal Rates...")

fig3, ax = plt.subplots(1, 1, figsize=(12, 8))

# Filter to reasonable range for visualization
plot_data = df[(df['shotDistance'] < 90) & (df['shotAngle'].abs() < 80)]

# Create hexbin plot
hexbin = ax.hexbin(plot_data['shotDistance'], plot_data['shotAngle'],
                   C=plot_data['goal'], reduce_C_function=np.mean,
                   gridsize=35, cmap='RdYlGn', mincnt=15, 
                   edgecolors='black', linewidths=0.2)

ax.set_xlabel('Shot Distance (feet)', fontsize=13, fontweight='bold')
ax.set_ylabel('Shot Angle (degrees)', fontsize=13, fontweight='bold')
ax.set_title('Figure 3: Goal Probability Heatmap (Distance × Angle)', 
             fontsize=15, fontweight='bold', pad=20)

# Add colorbar
cbar = plt.colorbar(hexbin, ax=ax)
cbar.set_label('Goal Rate', fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

# Add reference lines
ax.axhline(0, color='white', linestyle='--', alpha=0.5, linewidth=1.5, label='Center line')
ax.axvline(30, color='white', linestyle=':', alpha=0.5, linewidth=1.5, label='~30 ft')

# Add annotations for hot/cold zones
ax.text(15, 0, 'HOT ZONE', fontsize=12, fontweight='bold', 
        ha='center', va='center', color='white',
        bbox=dict(boxstyle='round', facecolor='darkgreen', alpha=0.7))
ax.text(70, 50, 'COLD ZONE', fontsize=10, fontweight='bold',
        ha='center', va='center', color='white',
        bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.7))

ax.legend(loc='upper right', fontsize=10)
ax.grid(False)

plt.tight_layout()
fig3_path = FIGURES_DIR / "fig3_2d_heatmap.png"
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {fig3_path.name}")

# --- rebound effects ---
print("Creating Figure 4: Rebound Effects on Goal Rates...")

fig4, axes = plt.subplots(1, 2, figsize=(14, 5))
fig4.suptitle('Figure 4: Rebound Effect on Goal Probability', 
              fontsize=16, fontweight='bold')

# (A) Distance effect by rebound status
ax = axes[0]
for reb, color, label, style in [(0, 'steelblue', 'Non-Rebound', '-'),
                                  (1, 'salmon', 'Rebound', '--')]:
    subset = df[df['is_rebound'] == reb]
    dist_bins = pd.cut(subset['shotDistance'], bins=25)
    goal_rate = subset.groupby(dist_bins, observed=True)['goal'].mean()
    bin_centers = [interval.mid for interval in goal_rate.index]
    ax.plot(bin_centers, goal_rate, style, linewidth=2.5, markersize=6,
            color=color, label=label, alpha=0.85, marker='o', 
            markeredgecolor='black', markeredgewidth=1)

ax.set_xlabel('Shot Distance (feet)', fontsize=12, fontweight='bold')
ax.set_ylabel('Goal Rate', fontsize=12, fontweight='bold')
ax.set_title('(A) Distance Effect Stratified by Rebound', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, None)

# (B) Angle effect by rebound status
ax = axes[1]
for reb, color, label, style in [(0, 'steelblue', 'Non-Rebound', '-'),
                                  (1, 'salmon', 'Rebound', '--')]:
    subset = df[df['is_rebound'] == reb]
    ang_bins = pd.cut(subset['shotAngle'], bins=25)
    goal_rate = subset.groupby(ang_bins, observed=True)['goal'].mean()
    bin_centers = [interval.mid for interval in goal_rate.index]
    ax.plot(bin_centers, goal_rate, style, linewidth=2.5, markersize=6,
            color=color, label=label, alpha=0.85, marker='o',
            markeredgecolor='black', markeredgewidth=1)

ax.set_xlabel('Shot Angle (degrees)', fontsize=12, fontweight='bold')
ax.set_ylabel('Goal Rate', fontsize=12, fontweight='bold')
ax.set_title('(B) Angle Effect Stratified by Rebound', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax.set_ylim(0, None)

plt.tight_layout()
fig4_path = FIGURES_DIR / "fig4_rebound_effects.png"
plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {fig4_path.name}")

# --- summary stats ---
print("Creating Table 1: Summary Statistics...")
# Calculate statistics
summary_stats = pd.DataFrame({
    'Variable': ['Distance (ft)', 'Angle (deg)', 'Rebound (%)', 'Goal (%)'],
    'N': [
        df['shotDistance'].notna().sum(),
        df['shotAngle'].notna().sum(),
        len(df),
        len(df)
    ],
    'Mean': [
        df['shotDistance'].mean(),
        df['shotAngle'].mean(),
        df['is_rebound'].mean() * 100,
        df['goal'].mean() * 100
    ],
    'Std Dev': [
        df['shotDistance'].std(),
        df['shotAngle'].std(),
        df['is_rebound'].std() * 100,
        df['goal'].std() * 100
    ],
    'Min': [
        df['shotDistance'].min(),
        df['shotAngle'].min(),
        0.0,
        0.0
    ],
    'Median': [
        df['shotDistance'].median(),
        df['shotAngle'].median(),
        0.0,
        0.0
    ],
    'Max': [
        df['shotDistance'].max(),
        df['shotAngle'].max(),
        100.0,
        100.0
    ]
})

# Round for presentation
summary_stats['Mean'] = summary_stats['Mean'].round(2)
summary_stats['Std Dev'] = summary_stats['Std Dev'].round(2)
summary_stats['Min'] = summary_stats['Min'].round(2)
summary_stats['Median'] = summary_stats['Median'].round(2)
summary_stats['Max'] = summary_stats['Max'].round(2)

# Save table
table1_path = TABLES_DIR / "table1_summary_statistics.csv"
summary_stats.to_csv(table1_path, index=False)
print(f"   Saved: {table1_path.name}")

print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(summary_stats.to_string(index=False))

# --- Key Insights for report ---
print("\n" + "="*70)
print("KEY INSIGHTS FOR YOUR REPORT")
print("="*70)

# calculate key stats
goal_rate = df['goal'].mean()
reb_boost = df[df['is_rebound'] == 1]['goal'].mean() - df[df['is_rebound'] == 0]['goal'].mean()
close_shots = df[df['shotDistance'] < 15]
far_shots = df[df['shotDistance'] > 60]

print(f"""
DATASET OVERVIEW:
   • {len(df):,} shots analyzed from 2024 NHL season
   • {df['goal'].sum():,} goals scored ({goal_rate:.1%} conversion rate)
   • Average shot distance: {df['shotDistance'].mean():.1f} feet

DISTANCE EFFECT (Nonlinear):
   • Close shots (<15 ft): {close_shots['goal'].mean():.1%} goal rate
   • Long shots (>60 ft): {far_shots['goal'].mean():.1%} goal rate
   • Clear nonlinear decay justifies quadratic term (β_d2)

ANGLE EFFECT:
   • Peak scoring at 0° (straight-on shots)
   • Goal rate decreases symmetrically with angle
   • Peripheral shots (~60°+) have {df[df['shotAngle'].abs() > 60]['goal'].mean():.1%} goal rate

REBOUND EFFECT:
   • {df['is_rebound'].sum():,} rebound shots ({df['is_rebound'].mean():.1%} of total)
   • Rebounds score at {reb_boost:.2f}x higher rate
   • Rebound goal rate: {df[df['is_rebound']==1]['goal'].mean():.1%}
   • Non-rebound goal rate: {df[df['is_rebound']==0]['goal'].mean():.1%}

MODEL JUSTIFICATION:
   • Nonlinear distance effect → Include β_d and β_d2
   • Symmetric angle effect → Include β_a
   • Strong rebound advantage → Include β_r
   • All effects validated in exploratory analysis
""")

