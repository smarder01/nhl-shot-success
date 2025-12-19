# NHL Shot Analysis - EDA Notes

**Project:** Baysian Modeling of NHL Shot Success
**Data:** 2024 NHL Season (119,870 shots)
**Goal:** Predict goal probability from distance, angle, and rebound status
**Project:** November 2024

---

## Dataset Overview

Basic Statistics
    - Total shots analyzed: 119,870
    - Total goals: 8,428 (7.0% conversion rate)
    - Rebounds: 8,937 shots (7.5% of total)
    - Data quality: No missing values in key variables

Key Variables
| Variable       | Mean | Std Dev | Min    | Max    |
|----------------|------|---------|--------|--------|
| Distance (ft)  | 34.2 | 19.9    | 1.0    | 98.4   |
| Angle (deg)    | 0.5  | 40.8    | -88.5  | +88.5  |
| Rebound (%)    | 7.5% | —       | —      | —      |
| Goal (%)       | 7.0% | —       | —      | —      |

---

## Figure 1: Univariate Distributions

Panel A: Shot Distance Distribution

What it shows:
- Distribution of where shots are taken from
- Bimodal pattern: peaks around 10-15 ft and 30-40 ft

Key insights:
- Mean distance: 34.2 ft, Median: 32.8 ft
- Most shots from 10-50 feet (typical NHL scoring range)
- Second peak at ~30 ft represents "slot" shots (high-danger area)
- Long tail extending to 100 ft (low-percentage point shots)

Implications for modeling:
- Wide range of distances requires flexible model
- Nonlinear relationship expected (visible in Figure 2)

### Panel B: Shot Angle Distribution
**What it shows:**
- Distribution of shot angles relative to net center
- Negative = left side, Positive = right side, 0° = straight-on

**Key insights:**
- Nearly symmetric around 0° (center)
- Mean: 0.5° (essentially centered)
- Two peaks around ±30° (offensive zone face-off dots)
- Drops off at extreme angles (±70°+)

**Implications for modeling:**
- Symmetric pattern suggests absolute value or even-powered terms
- Linear angle term (β_a) should capture penalty for peripheral shots
- No need for left/right distinction

### Panel C: Shot Outcomes
**What it shows:**
- Class imbalance in target variable

**Key insights:**
- 111,442 non-goals (93.0%)
- 8,428 goals (7.0%)
- 13:1 ratio of non-goals to goals

**Implications for modeling:**
- Highly imbalanced classification problem (typical for NHL)
- Model will naturally predict "no goal" most of the time
- Need to evaluate with appropriate metrics (not just accuracy)
- 7% baseline rate is what a naive model would achieve

### Panel D: Goal Rate by Rebound Status
**What it shows:**
- Direct comparison of scoring rates for rebounds vs non-rebounds

**Key insights:**
- **Non-rebounds:** 6.67% goal rate (n=110,933)
- **Rebounds:** 11.53% goal rate (n=8,937)
- **Rebound advantage: 1.73x** (73% increase in goal probability)

**Why rebounds score more:**
- Goalies out of position after first save
- Puck already near the net (shorter distance)
- Defenders scrambling, less able to block
- Less time for goalie to reset

**Implications for modeling:**
- Strong justification for including β_r (rebound parameter)
- Effect size (~5 percentage points) is substantial in hockey
- Should investigate if rebound effect varies by distance/angle (Figure 4)

---

## Figure 2: Bivariate Relationships

### Panel A: Nonlinear Distance Effect
**What it shows:**
- Goal rate as a function of shot distance
- Clear **nonlinear decay pattern**

**Key insights:**
1. **Close range (< 15 ft):** 
   - Goal rate: 13-32%
   - Steep drop-off from 32% → 13%
   
2. **Mid range (15-40 ft):**
   - Goal rate: 5-13%
   - Moderate decline
   
3. **Long range (40-100 ft):**
   - Goal rate: 1-5%
   - Flatter, slower decline

**Why this pattern exists:**
- Close: Goalie reaction time limited, shooting angle better
- Mid: Balance between difficulty and frequency
- Long: Low percentage but attempted when no better option

**Statistical interpretation:**
- **NOT a linear relationship** (doesn't drop at constant rate)
- Curve shows **concave shape** (steep then flat)
- This is the **primary justification** for including β_{d2} (quadratic term)

**Model formulation:**
```
logit(p) = α + β_d · distance + β_{d2} · distance²
```
- β_d < 0 (negative linear effect)
- β_{d2} < 0 (negative quadratic = concave down)
- Together they create the observed curve

---

### Panel B: Angle Effect (Symmetrical)
**What it shows:**
- Goal rate as a function of shot angle
- Peak at center (0°), declining toward periphery

**Key insights:**
- **Center shots (0°):** ~10% goal rate (highest)
- **Moderate angles (±30°):** ~6-8% goal rate
- **Extreme angles (±70°+):** ~3-5% goal rate (lowest, except behind net)

**Why this pattern exists:**
- Straight-on shots: Full net visible to shooter, goalie covers less
- Peripheral shots: Net "shrinks" from shooter's perspective
- Behind net angles (±85°+): Wrap-arounds, very difficult

**Statistical interpretation:**
- Approximately **symmetric** around 0°
- Left vs right doesn't matter (confirms we don't need separate terms)
- Relationship appears roughly linear in absolute angle
- Some noise at extremes due to smaller sample sizes

**Model formulation:**
```
logit(p) = ... + β_a · angle
```
- β_a should capture the penalty for moving away from center
- Sign depends on whether we use absolute angle or raw angle

---

## Figure 3: 2D Heatmap (Distance × Angle)

### What it shows:
- Joint distribution of goal probability across the offensive zone
- Hexagonal bins colored by empirical goal rate
- White areas = insufficient data (< 15 shots)

### Visual interpretation:
- **Green/yellow (HOT ZONE):** Close to net, centered (0-20 ft, ±20°)
  - Goal rates: 15-80% (some cells very high due to small samples)
  - This is the "slot" or "home plate" area in hockey
  
- **Orange/yellow (WARM ZONE):** Mid-range, moderate angles (20-40 ft, ±30°)
  - Goal rates: 8-15%
  - Still dangerous but more difficult
  
- **Red (COLD ZONE):** Long distance or extreme angles
  - Goal rates: 0-5%
  - Point shots, perimeter shots

### Key insights:
1. **Hot zone is small:** Elite scoring area is ~15 ft × 40° cone
2. **Gradient pattern:** Smooth transition from hot → cold (validates continuous model)
3. **Symmetry confirmed:** Left and right sides mirror each other
4. **Distance dominates angle:** Moving back 20 ft worse than moving 30° sideways

### Implications for modeling:
- Confirms both distance and angle matter
- No obvious interaction needed (effects appear additive)
- Could consider interaction term if model fit poor
- This heatmap is your **predicted probability surface**

### For your report:
- This is a **great visual** for showing model predictions later
- Compare observed heatmap (this one) to model-predicted heatmap
- Shows practical implications: "where should players shoot from?"

---

## Figure 4: Rebound Effects

### Panel A: Distance Effect Stratified by Rebound
**What it shows:**
- Blue line: Non-rebounds
- Red dashed line: Rebounds
- Same distance, different outcomes

**Key insights:**
1. **At close range (< 15 ft):**
   - Rebounds and non-rebounds have similar rates (~25-30%)
   - Rebound advantage minimal when already very close
   - Explanation: Already so close that rebound doesn't help much more
   
2. **At mid-range (15-40 ft):**
   - **Biggest rebound advantage**
   - Rebounds: ~10-12% goal rate
   - Non-rebounds: ~5-8% goal rate
   - Rebounds essentially "move" shots closer to net
   
3. **At long range (40+ ft):**
   - Both very low (~2-5%)
   - Rebounds rare at this distance (hard to rebound a point shot)
   - When they occur, small advantage

**Statistical interpretation:**
- Rebound effect appears **constant across distances** (parallel lines)
- Suggests **additive model** is appropriate: β_d + β_r (no interaction)
- Simplifies interpretation: rebound adds ~5% regardless of distance

---

### Panel B: Angle Effect Stratified by Rebound
**What it shows:**
- Same pattern for angle
- Red (rebounds) consistently above blue (non-rebounds)

**Key insights:**
1. **Peak at center (0°) for both groups:**
   - Rebounds: ~15% goal rate
   - Non-rebounds: ~9% goal rate
   - 6 percentage point boost
   
2. **Advantage maintained at all angles:**
   - Rebound line stays above non-rebound line
   - Roughly parallel pattern (additive effect)
   
3. **More noise in rebound line:**
   - Fewer rebounds overall (only 8,937)
   - Especially at extreme angles (rare situations)

**Statistical interpretation:**
- Again suggests **no interaction** needed
- Rebound effect doesn't depend on angle
- Additive model: β_a + β_r

---

## Model Justification Summary

Based on EDA, our model specification is:

```
goal_i ~ Bernoulli(p_i)
logit(p_i) = α + β_d · distance_i + β_{d2} · distance_i² + β_a · angle_i + β_r · rebound_i
```

### Why each parameter is justified:

| Parameter | Justification | Evidence |
|-----------|---------------|----------|
| **α** | Baseline log-odds | 7% overall goal rate |
| **β_d** | Linear distance effect | Fig 2A: clear decline |
| **β_{d2}** | Nonlinear distance effect | Fig 2A: steep then flat (concave) |
| **β_a** | Angle effect | Fig 2B: peak at center, decline at periphery |
| **β_r** | Rebound effect | Fig 1D, Fig 4: 1.73x advantage |

### Why we DON'T need:
- **Distance × Angle interaction:** Fig 3 shows smooth gradient (no "special" zones)
- **Distance × Rebound interaction:** Fig 4A shows parallel lines
- **Angle × Rebound interaction:** Fig 4B shows parallel lines
- **Left vs Right side terms:** Fig 2B, Fig 3 show symmetry

### Expected parameter signs:
- **α**: Could be positive or negative (depends on standardization)
- **β_d**: **Negative** (more distance = lower probability)
- **β_{d2}**: **Negative** (concave down curve)
- **β_a**: **Negative** (absolute angle; moving from center reduces probability)
- **β_r**: **Positive** (rebounds increase probability)

---

## Data Quality Assessment

### Strengths:
Large sample size (119,870 shots)  
No missing values in key variables  
Reasonable ranges (1-98 ft, ±88°)  
Sufficient events (8,428 goals) for stable estimates  
Clear patterns in data (validates model structure)

### Limitations:
Class imbalance (93% non-goals) - typical for hockey  
Some noise at extremes (long distance, extreme angles)  
Omitted variables: shot type, goalie quality, traffic, game state  
Measurement error in coordinates (play-by-play data can be noisy)

### Data cleaning performed:
- Removed 0 rows with missing values (none existed)
- Removed invalid distances (< 1 ft or > 100 ft): minimal
- Removed extreme angles (> ±89°, behind net): minimal
- **Final dataset: ~119,870 shots retained (~100%)**

---

## Standardization Details

For Bayesian modeling, we standardized continuous predictors:

### Distance:
```
distance_std = (distance - 34.22) / 19.85
```
- Mean = 34.22 ft
- SD = 19.85 ft
- Standardized mean ≈ 0, SD ≈ 1

### Angle:
```
angle_std = (angle - 0.52) / 40.79
```
- Mean = 0.52°
- SD = 40.79°
- Standardized mean ≈ 0, SD ≈ 1

### Why standardize?
1. **Prior selection:** Normal(0, 1) priors are reasonable for standardized predictors
2. **Numerical stability:** Similar scales prevent overflow/underflow
3. **Interpretation:** One SD change in predictor is meaningful unit
4. **Comparison:** Can compare β_d vs β_a magnitudes fairly

### Interpreting standardized coefficients:
- β_d = -0.5 means: "1 SD increase in distance (19.85 ft) decreases log-odds by 0.5"
- Can convert back to original scale: effect per foot = β_d / 19.85