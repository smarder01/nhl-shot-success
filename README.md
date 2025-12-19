# NHL Shot Success Prediction using Bayesian Logistic Regression

**Author:** Sydney Marder  
**Tools:** Python, PyMC, ArviZ, scikit-learn, pandas, matplotlib  
**Domain:** Sports Analytics · Bayesian Modeling · Applied Statistics

---

## TL;DR

- Built a **Bayesian logistic regression model** to predict NHL goal probability  
- Modeled **nonlinear distance effects** and shot context (angle, rebounds)  
- Used **PyMC (NUTS sampler)** with full convergence diagnostics  
- Achieved **AUC ≈ 0.71** with interpretable posterior estimates  
- Demonstrates an **end-to-end statistical modeling workflow**: EDA → baseline → Bayesian inference → interpretation  

---

## Project Overview

This project models the probability that an NHL shot results in a goal using **Bayesian logistic regression**.  
Using **119,870 shots from the 2024 NHL season**, the model estimates how shot distance, shot angle, and rebound status influence scoring probability while explicitly quantifying uncertainty.

The project emphasizes:
- interpretable modeling
- principled prior selection
- diagnostic-driven validation
- comparison between frequentist and Bayesian approaches

---

## Research Questions

- How does **shot distance** affect goal probability, and is the effect nonlinear?
- Does **shot angle** matter after controlling for distance?
- How much additional value do **rebound shots** provide once location is accounted for?
- How do Bayesian estimates compare to a standard MLE logistic regression baseline?

---

## Dataset

**Source:** MoneyPuck NHL Shot Data  
**Season:** 2024 NHL Regular Season  

- **Total shots:** 119,870  
- **Goals:** 8,428 (7.0% conversion rate)

### Variables Used

| Variable | Description |
|--------|------------|
| `shotDistance` | Distance from net (feet) |
| `shotAngle` | Angle from center ice (degrees) |
| `shotRebound` | Binary rebound indicator |
| `goal` | Binary outcome (0/1) |

### Data Quality

- No missing values in key variables  
- Realistic value ranges (distance: 1–98 ft, angle: ±88°)  
- Large sample size supports stable inference  

---

## Statistical Model

### Model Specification

$$
\text{goal}_i \sim \text{Bernoulli}(p_i)
$$

$$
\text{logit}(p_i) =
\alpha
+ \beta_d \cdot \text{distance}_i
+ \beta_{d2} \cdot \text{distance}_i^2
+ \beta_a \cdot \text{angle}_i
+ \beta_r \cdot \text{rebound}_i
$$

### Parameters

| Parameter | Interpretation |
|---------|----------------|
| α | Baseline log-odds of scoring |
| β_d | Linear distance effect |
| β_{d2} | Nonlinear distance curvature |
| β_a | Shot angle effect |
| β_r | Rebound effect |

### Priors (Weakly Informative)

```python
α ~ Normal(-2.5, 1.0)
β_d ~ Normal(-0.5, 0.5)
β_{d2} ~ Normal(-0.1, 0.3)
β_a ~ Normal(-0.2, 0.3)
β_r ~ Normal(+0.5, 0.3)
```

---

## Modeling Workflow

1. **Exploratory Data Analysis**
      - Univariate shot distributions
      - Goal rate vs distance and angle
      - Rebound stratification
      - Shot-location heatmaps
2. **Baseline Model**
      - MLE logistic regression (scikit-learn)
      - Used as a performance benchmark
3. **Bayesian Model**
      - PyMC implementation using NUTS
      - Posterior inference and uncertainty quantification
4. **Diagnostics & Validation**

---

## Key Results

### Exploratory Findings

**Distance:**
   - < 15 ft → ~13% goal rate
   - 60 ft → ~2% goal rate
   - Strong nonlinear decay
**Angle:**
   - Central shots outperform wide-angle shots
   - Symmetric left/right pattern
**Rebounds (Univariate):**
   - 11.5% goal rate vs 6.7% for non-rebounds

### Baseline MLE Performance

- AUC-ROC: 0.706
- Brier Score: 0.063

### Bayesian Model Results

**Convergence**
- 4 chains, NUTS sampler
- $\hat{R}$ = 1.0 for all parameters
- Effective sample size > 3,600
- No divergences

### Posterior Estimates
| Parameter | Mean | 95% Credible Interval | Odds Ratio |
|----------|------|-----------------------|------------|
| α | -2.81 | [-2.84, -2.78] | 0.06 |
| β_d (Distance) | -0.78 | [-0.80, -0.75] | 0.46 |
| $\beta_{d2}$ (Distance²) | +0.06 | [+0.03, +0.08] | 1.06 |
| β_a (Angle) | -0.29 | [-0.31, -0.27] | 0.75 |
| β_r (Rebound) | -0.03 | [-0.10, +0.05] | 0.97 |

---

## Modeling Insights
- Distance is the dominant driver of goal probability
- Angle retains a meangful independent effect
- Rebound advantage largely disappears once location is controlled for
- Ptositive quadratic term likely reflects multicollinearity
- Demonstrates the importance of multivariate vs unicariate analysis

---

## Project Structure
nhl-shot-modeling/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── 01_initial_exploration.py
│   ├── 02_comprehensive_eda.py
│   ├── 03_data_cleaning_preprocessing.py
│   ├── 04_baseline_model.py
│   ├── 05_pymc_bayesian_model.py
│   └── 06_diagnostics_visualization.py
├── figures/
├── tables/
├── models/
│   └── pymc_trace.nc
├── notebooks/
│   ├── eda_notes.md
│   └── baseline_model_notes.md
├── README.md
└── requirements.txt

---

## Skills Demonstrated
- Bayesian inference with PyMC (NUTS)
- Logistic regression (MLE vs Bayesian comparison)
- Statistical modeling and uncertainty quantification
- Exploratory data analysis and feature engineering
- Model diagnostics and posterior predictive checks
- Sports analytics and expected-goals modeling
- Reproducible research practices

---

## Limitations & Future Work
- Hierarchical models (player and team effects)
- Alternative nonllinear distance parameterizations
- Expanded expected-goals (xG) feature set
- Temporal and game-state effects

