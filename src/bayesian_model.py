"""
Main Bayesian analysis using PyMC3
Fit logistic regression model with MCMC sampling to obtain posterior distributions
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"

MODELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

print("="*70)
print("BAYESIAN LOGISTIC REGRESSION WITH PyMC3")
print("="*70)

# --- load data ---
print("Loading processed data...")

for filename in ['shots_analysis_ready_FINAL.csv', 'shots_analysis_ready_FIXED.csv', 
                 'shots_analysis_ready.csv']:
    filepath = DATA_DIR / filename
    if filepath.exists():
        df = pd.read_csv(filepath)
        print(f"   Loaded: {filename}")
        break

print(f"   Dataset: {len(df):,} shots")
print(f"   Goal rate: {df['goal'].mean():.3%}")

# prepare data
x_distance = df['distance_std'].values
x_distance_sq = df['distance_std_sq'].values

# use absolute angle if available, otherwise regular angle
if 'angle_abs_std' in df.columns:
    x_angle = df['angle_abs_std'].values
    print("Using: angle_abs_std")
elif 'angle_std' in df.columns:
    x_angle = df['angle_std'].values
    print("Using: angle_std")
else:
    raise ValueError("No angle column found in the dataset.")

x_rebound = df['is_rebound'].values
y = df['goal'].values

print(f"\n   Features ready:")
print(f"   Distance (standardized): mean={x_distance.mean():.4f}, std={x_distance.std():.4f}")
print(f"   Distance^2 (standardized): mean={x_distance_sq.mean():.4f}, std={x_distance_sq.std():.4f}")
print(f"   Angle (standardized): mean={x_angle.mean():.4f}, std={x_angle.std():.4f}")
print(f"   Rebound (binary): {x_rebound.sum():,} rebounds ({x_rebound.mean():.1%})")
print(f"   Goal (binary): {y.sum():,} goals ({y.mean():.3%})")

# --- pymc3 model specification ---
print("Building PyMC3 model ... ")

with pm.Model() as hockey_model:

    # Intercept: baseline log-odds
    # With 7% base rate, log(0.07/0.93) ≈ -2.6
    # Allow some flexibility around this
    alpha = pm.Normal('alpha', mu=-2.5, sigma=1.0)
    
    # Distance effect: expect negative (farther = worse)
    # From MLE we saw around -0.7 to -0.8
    beta_d = pm.Normal('beta_d', mu=-0.5, sigma=0.5)
    
    # Distance² effect: expect negative (concave curve)
    # Smaller magnitude than linear term
    beta_d2 = pm.Normal('beta_d2', mu=-0.1, sigma=0.3)
    
    # Angle effect: expect negative (peripheral = worse)
    # From MLE we saw around -0.3
    beta_a = pm.Normal('beta_a', mu=-0.2, sigma=0.3)
    
    # Rebound effect: expect positive (rebounds score more)
    # From EDA: 1.73x advantage → log(1.73) ≈ 0.55
    beta_r = pm.Normal('beta_r', mu=0.5, sigma=0.3)

# --- linear predictor (log-odds)
    logit_p = (alpha + beta_d * x_distance + beta_d2 * x_distance_sq + beta_a * x_angle + beta_r * x_rebound)

# --- likelihood ---
# bernoulli dist for binary outcomes
    y_obs = pm.Bernoulli('y_obs', logit_p=logit_p, observed=y)
    
    print("\n   Model specification:")
    print("   Parameters: alpha, beta_d, beta_d2, βbeta_a, beta_r")
    print("   Priors: Normal distributions (weakly informative)")
    print("   Likelihood: Bernoulli(logit_p)")
    print("   Total parameters: 5")

# --- mcmc sampling ---
print("\n   Running MCMC sampling...")
print("   Settings:")
print("   Chains: 4")
print("   Samples per chain: 1000")
print("   Tuning samples: 500")
print("   Total posterior samples: 4000")

with hockey_model:
    # sample from posterior
    trace = pm.sample(
        draws=1000,
        tune = 500,
        chains=4,
        cores=4,
        target_accept=0.90,
        return_inferencedata=True,
        random_seed=42,
        progressbar=True
    )

    print("sampling complete!")

# --- save trace ---
print("Saving results...")
trace_path = MODELS_DIR / "pymc3_trace.nc"
trace.to_netcdf(trace_path)
print(f"Trace saved: {trace_path.name}")

print("quick diagnostics ... ")

# Convergence summary
summary = az.summary(trace, var_names=['alpha', 'beta_d', 'beta_d2', 'beta_a', 'beta_r'])
print("\n" + "="*70)
print("POSTERIOR SUMMARY")
print("="*70)
print(summary.to_string())

# Check convergence
print("\n" + "="*70)
print("CONVERGENCE DIAGNOSTICS")
print("="*70)

rhat_check = (summary['r_hat'] < 1.01).all()
ess_check = (summary['ess_bulk'] > 400).all()

print(f"\n   R-hat < 1.01: {'PASS' if rhat_check else 'FAIL'}")
print(f"   ESS > 400:    {'PASS' if ess_check else 'FAIL'}")

if rhat_check and ess_check:
    print("\n   Model converged successfully!")
else:
    print("\n   Convergence issues detected. May need more samples.")


# --- parameter interpretation ---
print("\n" + "="*70)
print("PARAMETER INTERPRETATION")
print("="*70)

posterior_means = summary['mean']

print(f"\n   Posterior Means (95% Credible Intervals):")
for param in ['alpha', 'beta_d', 'beta_d2', 'beta_a', 'beta_r']:
    mean = summary.loc[param, 'mean']
    lower = summary.loc[param, 'hdi_3%']
    upper = summary.loc[param, 'hdi_97%']
    print(f"   • {param:10s}: {mean:+.4f}  [{lower:+.4f}, {upper:+.4f}]")

print(f"\n   Odds Ratios:")
print(f"   Distance (1 SD = 19.85 ft): {np.exp(posterior_means['beta_d']):.3f}")
print(f"   Angle (1 SD = ~25°):        {np.exp(posterior_means['beta_a']):.3f}")
print(f"   Rebound effect:             {np.exp(posterior_means['beta_r']):.3f}")
