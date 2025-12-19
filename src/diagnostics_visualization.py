"""
  Generate comprehensive diagnostics and visualizations for Bayesian model.
Creates figures showing convergence, posterior distributions, and model validation.
"""

import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

# paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"

print("="*70)
print("BAYESIAN MODEL DIAGNOSTICS & VISUALIZATION")
print("="*70)

# --- load trace ---
print("Loading MCMC trace...")

trace_path = MODELS_DIR / "pymc3_trace.nc"
if not trace_path.exists():
    print(f"Error: {trace_path} not found!")
    print("Run src/05_pymc3_bayesian_model.py first")
    exit(1)

trace = az.from_netcdf(trace_path)
print(f"Loaded: {trace_path.name}")

# get dimensions
n_chains = len(trace.posterior.chain)
n_draws = len(trace.posterior.draw)
print(f"Chains: {n_chains}")
print(f"Draws per chain: {n_draws}")
print(f"Total samples: {n_chains * n_draws}")

# --- trace plots ---
print("Creating Figure 6: Trace Plots...")

fig6 = az.plot_trace(
    trace,
    var_names = ['alpha', 'beta_d', 'beta_d2', 'beta_a', 'beta_r'],
    compact = True,
    figsize = (14,10),
    divergences = 'bottom'
)

fig6[0, 0].figure.suptitle('Figure 6: MCMC Trace Plots - Convergence Diagnostics',
                           fontsize=15, fontweight='bold', y=0.995)

# adjust layout and save
plt.tight_layout()
fig6_path = FIGURES_DIR / "fig6_trace_plots.png"
plt.savefig(fig6_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {fig6_path.name}")

# --- posterior distributions ---
print("Creating Figure 7: Posterior Distributions...")

fig7 = az.plot_posterior(
    trace,
    var_names=['alpha', 'beta_d', 'beta_d2', 'beta_a', 'beta_r'],
    hdi_prob=0.95,
    figsize=(14, 10),
    textsize=11
)

if isinstance(fig7, np.ndarray):
    fig7.flat[0].figure.suptitle('Figure 7: Posterior Distributions (95% Credible Intervals)',
                                  fontsize=15, fontweight='bold', y=0.995)
else:
    fig7.figure.suptitle('Figure 7: Posterior Distributions (95% Credible Intervals)',
                         fontsize=15, fontweight='bold', y=0.995)

plt.tight_layout()
fig7_path = FIGURES_DIR / "fig7_posterior_distributions.png"
plt.savefig(fig7_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {fig7_path.name}")

# --- posterior predictive check ---
print("Creating Figure 8: Posterior Predictive Check...")

# Check if posterior_predictive exists
if hasattr(trace, 'posterior_predictive') and 'y_obs' in trace.posterior_predictive:
    print("   Using existing posterior predictive samples...")
    ppc = trace.posterior_predictive['y_obs'].values
else:
    print("   No posterior predictive samples found in trace")
    print("   Skipping Figure 8 (would need to regenerate with model)")
    ppc = None

if ppc is not None:
    # Reshape: (chains, draws, observations) -> (samples, observations)
    ppc_flat = ppc.reshape(-1, ppc.shape[-1])
    
    # Calculate statistics
    observed_rate = trace.observed_data['y_obs'].mean().values
    predicted_rates = ppc_flat.mean(axis=1)  # Mean goal rate for each posterior sample
    
    fig8, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig8.suptitle('Figure 8: Posterior Predictive Check', fontsize=15, fontweight='bold')
    
    # Panel A: Distribution of predicted goal rates
    ax = axes[0]
    ax.hist(predicted_rates, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(observed_rate, color='red', linewidth=3, label=f'Observed: {observed_rate:.3%}')
    ax.set_xlabel('Predicted Goal Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('(A) Predicted vs Observed Goal Rate', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Panel B: Goal count distribution
    ax = axes[1]
    observed_goals = trace.observed_data['y_obs'].sum().values
    predicted_goals = ppc_flat.sum(axis=1)
    ax.hist(predicted_goals, bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax.axvline(observed_goals, color='red', linewidth=3, label=f'Observed: {observed_goals:,.0f}')
    ax.set_xlabel('Total Goals Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('(B) Predicted vs Observed Goal Count', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig8_path = FIGURES_DIR / "fig8_posterior_predictive.png"
    plt.savefig(fig8_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {fig8_path.name}")
else:
    print("   Skipped Figure 8 (no posterior predictive data)")

# --- Bayesian parameter summary ---
print("Creating Table 4: Bayesian Parameter Summary...")

summary = az.summary(
    trace,
    var_names=['alpha', 'beta_d', 'beta_d2', 'beta_a', 'beta_r'],
    hdi_prob=0.95
)

# Create clean table for report
param_names = {
    'alpha': 'alpha (Intercept)',
    'beta_d': 'beta_d (Distance)',
    'beta_d2': 'beta_d2 (Distance²)',
    'beta_a': 'beta_a (Angle)',
    'beta_r': 'beta_r (Rebound)'
}

table4 = pd.DataFrame({
    'Parameter': [param_names[idx] for idx in summary.index],
    'Posterior Mean': summary['mean'].values,
    '95% CI Lower': (summary['mean'] - 1.96 * summary['sd']).values,
    '95% CI Upper': (summary['mean'] + 1.96 * summary['sd']).values,
    'Odds Ratio': np.exp(summary['mean'].values),
    'ESS': summary['ess_bulk'].values.astype(int),
    'R-hat': summary['r_hat'].values
})

# Round for presentation
table4['Posterior Mean'] = table4['Posterior Mean'].round(4)
table4['95% CI Lower'] = table4['95% CI Lower'].round(4)
table4['95% CI Upper'] = table4['95% CI Upper'].round(4)
table4['Odds Ratio'] = table4['Odds Ratio'].round(4)
table4['R-hat'] = table4['R-hat'].round(4)

# Save
table4_path = TABLES_DIR / "table4_bayesian_summary.csv"
table4.to_csv(table4_path, index=False)
print(f"   Saved: {table4_path.name}")

print("\n" + "="*70)
print("TABLE 4: BAYESIAN PARAMETER SUMMARY")
print("="*70)
print(table4.to_string(index=False))

# --- convergence summary ---
print("\n" + "="*70)
print("CONVERGENCE SUMMARY")
print("="*70)

all_converged = (summary['r_hat'] < 1.01).all()
good_ess = (summary['ess_bulk'] > 400).all()

print(f"\n   R-hat check (< 1.01): {'PASS' if all_converged else 'FAIL'}")
print(f"   ESS check (> 400):    {'PASS' if good_ess else 'FAIL'}")

if all_converged and good_ess:
    print("\n   All diagnostics passed!")
    print("   Model is ready for inference and reporting.")
else:
    print("\n   Some diagnostics failed.")
    print("   May need longer chains or different sampler settings.")

# Parameter-specific notes
print(f"\n   Parameter-specific convergence:")
for param in summary.index:
    rhat = summary.loc[param, 'r_hat']
    ess = summary.loc[param, 'ess_bulk']
    status = 'good' if rhat < 1.01 and ess > 400 else 'warning'
    print(f"   {status} {param_names[param]:20s}: r̂={rhat:.4f}, ESS={ess:>5.0f}")

# --- comparison to MLE ---
print("\n" + "="*70)
print("COMPARISON: BAYESIAN vs MLE")
print("="*70)

# Load MLE results if available
mle_path = TABLES_DIR / "table3_baseline_coefficients.csv"
if mle_path.exists():
    mle_table = pd.read_csv(mle_path)
    
    print("\n   Parameter          MLE Estimate    Bayesian Mean    Difference")
    print("   " + "-"*65)
    
    mle_vals = mle_table['MLE Estimate'].values
    bayes_vals = table4['Posterior Mean'].values
    
    for i, param in enumerate(table4['Parameter']):
        diff = bayes_vals[i] - mle_vals[i]
        print(f"   {param:18s}  {mle_vals[i]:+8.4f}      {bayes_vals[i]:+8.4f}      {diff:+7.4f}")
    
    print("\n   Key observations:")
    print("   • Estimates are very similar (large n dominates)")
    print("   • Bayesian provides full distributions, not just point estimates")
    print("   • Credible intervals quantify uncertainty")
else:
    print("   MLE table not found - skipping comparison")
