"""
07_predictive_uncertainty_plot.py

Create predictive plots showing uncertainty bands from Bayesian posterior.
Similar to assignment example: plot predictions with credible intervals.

Author: [Your Name]
Course: ISC 5228 - MCMC
Date: December 2024
"""

import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.special import expit  # logistic function
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"

print("="*70)
print("PREDICTIVE PLOTS WITH UNCERTAINTY")
print("="*70)

# ============================================================================
# LOAD DATA AND TRACE
# ============================================================================
print("\nLoading data and trace...")

# Load data
df = pd.read_csv(DATA_DIR / "shots_analysis_ready_FINAL.csv")
print(f"   Loaded {len(df):,} shots")

# Load trace
trace = az.from_netcdf(MODELS_DIR / "pymc3_trace.nc")
print(f"   Loaded trace with {len(trace.posterior.chain) * len(trace.posterior.draw)} samples")

# Extract posterior samples
posterior = trace.posterior
alpha_samples = posterior['alpha'].values.flatten()
beta_d_samples = posterior['beta_d'].values.flatten()
beta_d2_samples = posterior['beta_d2'].values.flatten()
beta_a_samples = posterior['beta_a'].values.flatten()
beta_r_samples = posterior['beta_r'].values.flatten()

print(f"   Posterior samples: {len(alpha_samples)}")

# --- figure 8: predicted prob ws distance (with uncertainty)

print("\nCreating Figure 8: Predictions vs Distance with Uncertainty...")

fig8, axes = plt.subplots(1, 2, figsize=(14, 6))
fig8.suptitle('Figure 8: Predicted Goal Probability with Uncertainty Bands', 
              fontsize=15, fontweight='bold')

# Panel A: Non-rebounds
ax = axes[0]

# Create range of standardized distances
distance_range = np.linspace(-2, 2, 100)  # Standardized scale
distance_range_sq = distance_range ** 2

# For each posterior sample, compute predicted probability
predictions = []
for i in range(min(1000, len(alpha_samples))):  # Use 1000 samples for speed
    logit_p = (alpha_samples[i] + 
               beta_d_samples[i] * distance_range + 
               beta_d2_samples[i] * distance_range_sq + 
               beta_a_samples[i] * 0 +  # Mean angle (0 after standardization)
               beta_r_samples[i] * 0)   # Non-rebound
    p = expit(logit_p)  # Convert to probability
    predictions.append(p)

predictions = np.array(predictions)

# Calculate percentiles
mean_pred = predictions.mean(axis=0)
lower_95 = np.percentile(predictions, 2.5, axis=0)
upper_95 = np.percentile(predictions, 97.5, axis=0)
lower_50 = np.percentile(predictions, 25, axis=0)
upper_50 = np.percentile(predictions, 75, axis=0)

# Convert standardized distance to original scale for plotting
distance_mean = 34.22
distance_std = 19.85
distance_original = distance_range * distance_std + distance_mean

# Observed data (binned)
df_nonreb = df[df['is_rebound'] == 0]
dist_bins = pd.cut(df_nonreb['shotDistance'], bins=20)
observed = df_nonreb.groupby(dist_bins, observed=True)['goal'].agg(['mean', 'count'])
bin_centers = [interval.mid for interval in observed.index]

# Plot
ax.plot(distance_original, mean_pred, 'b-', linewidth=3, label='Posterior Mean', zorder=3)
ax.fill_between(distance_original, lower_95, upper_95, alpha=0.2, color='blue', 
                label='95% Credible Interval', zorder=1)
ax.fill_between(distance_original, lower_50, upper_50, alpha=0.4, color='blue', 
                label='50% Credible Interval', zorder=2)
ax.plot(bin_centers, observed['mean'], 'ro', markersize=8, markeredgecolor='black',
        markeredgewidth=1, label='Observed', zorder=4, alpha=0.7)

ax.set_xlabel('Shot Distance (feet)', fontsize=12, fontweight='bold')
ax.set_ylabel('Goal Probability', fontsize=12, fontweight='bold')
ax.set_title('(A) Non-Rebounds (angle=0°)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 0.25])

# Panel B: Rebounds
ax = axes[1]

# Predictions for rebounds
predictions_reb = []
for i in range(min(1000, len(alpha_samples))):
    logit_p = (alpha_samples[i] + 
               beta_d_samples[i] * distance_range + 
               beta_d2_samples[i] * distance_range_sq + 
               beta_a_samples[i] * 0 + 
               beta_r_samples[i] * 1)   # Rebound
    p = expit(logit_p)
    predictions_reb.append(p)

predictions_reb = np.array(predictions_reb)

mean_pred_reb = predictions_reb.mean(axis=0)
lower_95_reb = np.percentile(predictions_reb, 2.5, axis=0)
upper_95_reb = np.percentile(predictions_reb, 97.5, axis=0)
lower_50_reb = np.percentile(predictions_reb, 25, axis=0)
upper_50_reb = np.percentile(predictions_reb, 75, axis=0)

# Observed rebounds
df_reb = df[df['is_rebound'] == 1]
if len(df_reb) > 100:  # Only if we have enough rebounds
    dist_bins_reb = pd.cut(df_reb['shotDistance'], bins=15)
    observed_reb = df_reb.groupby(dist_bins_reb, observed=True)['goal'].agg(['mean', 'count'])
    bin_centers_reb = [interval.mid for interval in observed_reb.index]
    
    ax.plot(bin_centers_reb, observed_reb['mean'], 'ro', markersize=8, 
            markeredgecolor='black', markeredgewidth=1, label='Observed', zorder=4, alpha=0.7)

ax.plot(distance_original, mean_pred_reb, 'r-', linewidth=3, label='Posterior Mean', zorder=3)
ax.fill_between(distance_original, lower_95_reb, upper_95_reb, alpha=0.2, color='red', 
                label='95% Credible Interval', zorder=1)
ax.fill_between(distance_original, lower_50_reb, upper_50_reb, alpha=0.4, color='red', 
                label='50% Credible Interval', zorder=2)

ax.set_xlabel('Shot Distance (feet)', fontsize=12, fontweight='bold')
ax.set_ylabel('Goal Probability', fontsize=12, fontweight='bold')
ax.set_title('(B) Rebounds (angle=0°)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 0.25])

plt.tight_layout()
fig8_path = FIGURES_DIR / "fig8_predictive_uncertainty.png"
plt.savefig(fig8_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {fig8_path.name}")

# --- figure 9: predicted prob vs angle (with uncertainty) ---

print("\nCreating Figure 9: Predictions vs Angle with Uncertainty...")

fig9, ax = plt.subplots(1, 1, figsize=(10, 6))
fig9.suptitle('Figure 9: Goal Probability by Angle (distance=30 ft, non-rebound)', 
              fontsize=15, fontweight='bold')

# Create range of standardized angles
angle_mean = df['shotAngle'].abs().mean() if 'shotAngle' in df.columns else 0
angle_std_val = df['shotAngle'].abs().std() if 'shotAngle' in df.columns else 40.79
angle_range_std = np.linspace(-1.5, 1.5, 100)

# Fixed distance at 30 ft (standardized)
distance_30 = (30 - distance_mean) / distance_std
distance_30_sq = distance_30 ** 2

# Predictions
predictions_angle = []
for i in range(min(1000, len(alpha_samples))):
    logit_p = (alpha_samples[i] + 
               beta_d_samples[i] * distance_30 + 
               beta_d2_samples[i] * distance_30_sq + 
               beta_a_samples[i] * angle_range_std + 
               beta_r_samples[i] * 0)
    p = expit(logit_p)
    predictions_angle.append(p)

predictions_angle = np.array(predictions_angle)

mean_pred_angle = predictions_angle.mean(axis=0)
lower_95_angle = np.percentile(predictions_angle, 2.5, axis=0)
upper_95_angle = np.percentile(predictions_angle, 97.5, axis=0)
lower_50_angle = np.percentile(predictions_angle, 25, axis=0)
upper_50_angle = np.percentile(predictions_angle, 75, axis=0)

# Convert to original scale
angle_original = angle_range_std * angle_std_val + angle_mean

# Observed (binned by angle)
df_30ft = df[(df['shotDistance'] > 25) & (df['shotDistance'] < 35) & (df['is_rebound'] == 0)]
if 'angle_abs' in df_30ft.columns:
    angle_bins = pd.cut(df_30ft['angle_abs'], bins=15)
    observed_angle = df_30ft.groupby(angle_bins, observed=True)['goal'].agg(['mean', 'count'])
    bin_centers_angle = [interval.mid for interval in observed_angle.index if observed_angle.loc[interval, 'count'] > 10]
    observed_means = [observed_angle.loc[interval, 'mean'] for interval in observed_angle.index if observed_angle.loc[interval, 'count'] > 10]
    
    ax.plot(bin_centers_angle, observed_means, 'ro', markersize=8, 
            markeredgecolor='black', markeredgewidth=1, label='Observed', zorder=4, alpha=0.7)

ax.plot(angle_original, mean_pred_angle, 'b-', linewidth=3, label='Posterior Mean', zorder=3)
ax.fill_between(angle_original, lower_95_angle, upper_95_angle, alpha=0.2, color='blue', 
                label='95% Credible Interval', zorder=1)
ax.fill_between(angle_original, lower_50_angle, upper_50_angle, alpha=0.4, color='blue', 
                label='50% Credible Interval', zorder=2)

ax.set_xlabel('Shot Angle (degrees from center)', fontsize=12, fontweight='bold')
ax.set_ylabel('Goal Probability', fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 0.15])

plt.tight_layout()
fig9_path = FIGURES_DIR / "fig9_angle_uncertainty.png"
plt.savefig(fig9_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {fig9_path.name}")

# --- summary stats ---

print("\nSummary statistics...")

print(f"\n   Uncertainty quantification:")
print(f"     At 30 ft, center, non-rebound:")
print(f"     Mean probability: {mean_pred[50]:.3f}")
print(f"     95% CI: [{lower_95[50]:.3f}, {upper_95[50]:.3f}]")
print(f"     Width: {upper_95[50] - lower_95[50]:.3f}")

print(f"\n   At 50 ft, center, non-rebound:")
idx_50 = np.argmin(np.abs(distance_original - 50))
print(f"     Mean probability: {mean_pred[idx_50]:.3f}")
print(f"     95% CI: [{lower_95[idx_50]:.3f}, {upper_95[idx_50]:.3f}]")
print(f"     Width: {upper_95[idx_50] - lower_95[idx_50]:.3f}")