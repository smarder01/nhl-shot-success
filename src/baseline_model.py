"""
Fit baseline logistic regression model using sklearn
This provides maximum likelihood estimates (MLE) for comparison with Bayesian results.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                              f1_score, roc_auc_score, brier_score_loss, log_loss)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# aet up paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"
MODELS_DIR = PROJECT_ROOT / "models"

# create directories
FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

print("="*70)
print("BASELINE LOGISTIC REGRESSION MODEL")
print("="*70)

# --- load data ---
print("\nLoading analysis-ready data...")
df = pd.read_csv(DATA_DIR / "shots_analysis_ready_FINAL.csv")
print(f"   Dataset: {len(df):,} shots")
print(f"   Features: distance_std, distance_std_sq, angle_std, is_rebound")
print(f"   Target: goal ({df['goal'].sum():,} goals, {df['goal'].mean():.2%} rate)")


# --- prepare features ---
print("\nPreparing features...")

# feature matrix
x = df[["distance_std", "distance_std_sq", "angle_abs_std", "is_rebound"]].values
y = df["goal"].values

print(f"   Feature matrix shape: {x.shape}")
print(f"   Target vector shape: {y.shape}")
print(f"   Class Balance: {(y==0).sum():,} non-goals, {(y==1).sum():,} goals")


# --- fit logistic regression model ---
print("\nFitting logistic regression model...")

# fit model with L2 regularization (like weakly informative priors)
model = LogisticRegression(
    penalty = 'l2',
    C = 1.0,
    solver = 'lbfgs',
    max_iter = 1000,
    random_state = 42
)

model.fit(x, y)
print("   Model fitting complete.")

# extract coefficients
intercept = model.intercept_[0]
coefficients = model.coef_[0]

print(f"\n   Model coefficients (MLE):")
print(f"   Intercept (alpha):         {intercept:+.4f}")
print(f"   Distance (beta_d):        {coefficients[0]:+.4f}")
print(f"   Distance^2 (beta_d2):      {coefficients[1]:+.4f}")
print(f"   Angle (beta_a):           {coefficients[2]:+.4f}")
print(f"   Rebound (beta_r):         {coefficients[3]:+.4f}")    


# --- model predicitions ---
print("\nGenerating model predictions...")

# predicted probabilities
y_pred_proba = model.predict_proba(x)[:, 1]

# binary predictions (threshold = 0.5)
y_pred = (y_pred_proba >= 0.5).astype(int)

print(f"   Mean predicted probability: {y_pred_proba.mean():.4f}")
print(f"   Predicted goals (threshold=0.5): {y_pred.sum():,}")


# --- model evaluation ---
print("\nEvaluating model performance...")

# classification metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
auc = roc_auc_score(y, y_pred_proba)

# probabilistic metrics
brier = brier_score_loss(y, y_pred_proba)
logloss = log_loss(y, y_pred_proba)

print(f"\n   Classification Metrics:")
print(f"   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1 Score:  {f1:.4f}")
print(f"   AUC:       {auc:.4f}")

print(f"\n   Probabilistic Metrics:")
print(f"   Brier Score: {brier:.4f}")
print(f"   Log Loss:    {logloss:.4f}")

# create results table
results_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC", "Brier Score", "Log Loss"],
    "Value": [accuracy, precision, recall, f1, auc, brier, logloss]
})

# save results
results_path = TABLES_DIR / "table2_baseline_performance.csv"
results_df.to_csv(results_path, index=False)
print(f"\n   Saved: {results_path.name}")


# --- visualizations ---
print("\nGenerating visualizations...")

# figure: Model Evaluation Metrics (3 Panels)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Figure 5: Baseline Model Evaluation (Maximum Likelihood)', 
             fontsize=15, fontweight='bold')

# panel A: Calibration Curve
ax = axes[0]
fraction_of_positives, mean_predicted_value = calibration_curve(
    y, y_pred_proba, n_bins=20, strategy='uniform'
)
ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
ax.plot(mean_predicted_value, fraction_of_positives, 'o-', 
        linewidth=2, markersize=8, color='steelblue', label='Model')
ax.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
ax.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
ax.set_title('(A) Calibration Curve', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

# Panel B: Predicted vs Observed by Distance Bins
ax = axes[1]
df['pred_proba'] = y_pred_proba
dist_bins = pd.cut(df['shotDistance'], bins=20)
binned = df.groupby(dist_bins, observed=True).agg({
    'goal': 'mean',
    'pred_proba': 'mean'
})
bin_centers = [interval.mid for interval in binned.index]
ax.plot(bin_centers, binned['goal'], 'o-', linewidth=2, markersize=8,
        color='coral', label='Observed', markeredgecolor='black')
ax.plot(bin_centers, binned['pred_proba'], 's--', linewidth=2, markersize=8,
        color='steelblue', label='Predicted', markeredgecolor='black')
ax.set_xlabel('Shot Distance (feet)', fontsize=12, fontweight='bold')
ax.set_ylabel('Goal Rate', fontsize=12, fontweight='bold')
ax.set_title('(B) Predicted vs Observed by Distance', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel C: Residuals by Predicted Probability
ax = axes[2]
residuals = y - y_pred_proba
ax.scatter(y_pred_proba, residuals, alpha=0.1, s=5, color='steelblue')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
# Add smoothed mean
prob_bins = np.linspace(0, y_pred_proba.max(), 50)
bin_indices = np.digitize(y_pred_proba, prob_bins)
mean_residuals = [residuals[bin_indices == i].mean() 
                  for i in range(1, len(prob_bins)) if (bin_indices == i).sum() > 0]
valid_bins = [prob_bins[i] for i in range(1, len(prob_bins)) if (bin_indices == i).sum() > 0]
ax.plot(valid_bins, mean_residuals, 'r-', linewidth=3, label='Mean residual')
ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
ax.set_ylabel('Residual (Observed - Predicted)', fontsize=12, fontweight='bold')
ax.set_title('(C) Residual Plot', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig5_path = FIGURES_DIR / "fig5_baseline_evaluation.png"
plt.savefig(fig5_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   Saved: {fig5_path.name}")


# --- coefficient table ---
print("\nModel Coefficients (Maximum Likelood Estimates)...")

# create coefficient table
coef_table = pd.DataFrame({
    'Parameter': ['alpha (Intercept)', 'beta_d (Distance)', 'beta_d2 (Distance^2)', 
                  'beta_a (Angle)', 'beta_r (Rebound)'],
    'MLE Estimate': [intercept, coefficients[0], coefficients[1], 
                     coefficients[2], coefficients[3]],
    'Odds Ratio': [np.exp(intercept), np.exp(coefficients[0]), 
                   np.exp(coefficients[1]), np.exp(coefficients[2]), 
                   np.exp(coefficients[3])]
})

coef_table['MLE Estimate'] = coef_table['MLE Estimate'].round(4)
coef_table['Odds Ratio'] = coef_table['Odds Ratio'].round(4)

print("\n" + coef_table.to_string(index=False))

# Save coefficient table
coef_path = TABLES_DIR / "table3_baseline_coefficients.csv"
coef_table.to_csv(coef_path, index=False)
print(f"\nSaved: {coef_path.name}")