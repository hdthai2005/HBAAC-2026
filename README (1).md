# HBAAC Round 2 — Team TheKing Solution

## Overview
- **Task:** Forecast daily demand for 15,972 auto-parts SKUs over 56 days.
- **Metric:** WRMSSE (lower is better).
- **Best Public LB:** 0.48679

## Approach Summary

We built a hybrid pipeline combining:
1. **Global LightGBM** over top 1500 high-profit SKUs (Tweedie objective)
2. **Per-SKU auto-ensemble** selecting best of: baseline, LGBM, ETS, Croston,
   trend-aware methods (Holt linear, linear drift, base+drift)
3. **CV-derived calibration** at 3 weight-tier levels (top100, top101-500,
   top501-1500), tuned via held-out folds
4. **Final blend** combining the calibrated model with a conservative
   no-calibration variant to hedge against regime shift in evaluation period

### Why blend?
Diagnostic comparing the calibrated model (v13.2) vs no-calibration variant
(v13) showed that v13.2 increased predictions for top-100 SKUs by:
- +25% in validation period (F1-F28)
- +67% in evaluation period (F29-F56)

The evaluation boost amplifies because trend-aware methods extrapolate over
56 days. To protect against possible regime shift between Sep (validation
analog) and Oct (evaluation analog), we apply a 80/20 blend on evaluation
rows only — validation rows stay identical to v13.2.

## File Structure

```
.
├── README.md                                    # This file
├── train-2026-v13-2.ipynb                       # Main calibrated model
├── train-2026-v13-trendaware.ipynb              # Conservative variant (for blend input)
├── make_blend.py                                # Post-processing blend script
└── submission_final.csv                         # Final submission
```

## How to Reproduce

### Step 1 — Train main model (v13.2)
Open `train-2026-v13-2.ipynb` on Kaggle with competition dataset
`hbaac-round2` attached. Run all cells. This produces:
- `submission_v13_2_level_calibration.csv`

Runtime: ~45 minutes on Kaggle CPU.

### Step 2 — Train conservative variant (v13)
Open `train-2026-v13-trendaware.ipynb`. Run all cells. This produces:
- `submission_v13_trendaware.csv`

Runtime: ~40 minutes on Kaggle CPU.

### Step 3 — Generate final blend
Place both CSVs in the same directory as `make_blend.py`, then run:

```bash
python make_blend.py
```

This generates `submission_v13_4_B_80_20.csv` — our final submission.

The blend formula:
```
final.validation = v13.2.validation              # unchanged
final.evaluation = 0.80 * v13.2.evaluation + 0.20 * v13.evaluation
```

## Methodology Details

### Cross-validation
- 3 historical holdout folds at offsets [28, 56, 84] days from train end
- Each fold evaluates a 28-day forecast against actuals
- Per-SKU model selection: choose method with lowest median RMSSE across folds
- Safety margin: only override `base` method if competitor beats it by >1%

### Features used (Global LGBM)
- Lags: 7, 14, 28 days
- Rolling mean: 28-day
- Price: current + 28-day lag
- Calendar: day-of-week, day-of-month, month
- Vietnamese holidays: Tet, Hung King Day, solar holidays

### Trend-aware methods
- `holt_linear`: Holt linear trend (no seasonality)
- `linear_drift`: simple linear extrapolation from last 28 days
- `base_drift`: baseline + monthly drift adjustment

### Calibration (v13.2)
For top 1500 SKUs grouped by 3 weight tiers:
1. Aggregate actual vs predicted across CV folds per tier
2. Compute ratio = sum(actual) / sum(predicted)
3. Clip ratio to [0.97, 1.03], shrink toward 1.0 with strength 0.70
4. Multiply final predictions by tier-specific ratio

### Blend (v13.4)
The blend is justified by the observation that v13.2's evaluation period
predictions are 16% higher in total than v13's, primarily due to
trend-aware methods extrapolating across the longer horizon. Since the
evaluation period (Oct) differs in seasonality from the validation period
(Sep), we hedge against over-extrapolation.

## Things We Tried That Didn't Work

- **Direct multi-step LGBM** (v12.2): Increased Public LB to 0.527 vs v10's
  0.49 — multi-step training distributed errors across all 56 days rather
  than focusing on near-term accuracy.
- **Per-SKU specialist for top-2 SKUs**: Tested baseline_56, baseline_28,
  YoY-naive, YoY-blend on Sep 2024 holdout. None improved over the
  auto-ensemble's selection for top-2 SKUs.
- **Stronger calibration (v13.3.1, +4% boost val)**: Public LB regressed
  to 0.48834 — calibration past v13.2's level over-corrects.

## Hardware / Runtime
- Kaggle Notebook, CPU + GPU
- LightGBM training: ~45 min for full pipeline
- Total reproduction time: ~90 min (both notebooks) + ~10s (blend)
