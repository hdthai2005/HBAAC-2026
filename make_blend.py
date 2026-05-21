
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
V132_CSV = Path("/mnt/user-data/uploads/submission_v13_2_level_calibration.csv")
V13_CSV  = Path("/mnt/user-data/uploads/submission_v13_trendaware.csv")
OUT = Path("/mnt/user-data/outputs")
OUT.mkdir(exist_ok=True)

# Load
v132 = pd.read_csv(V132_CSV)
v13  = pd.read_csv(V13_CSV)
F_cols = [f'F{i+1}' for i in range(28)]

# Sanity
assert len(v132) == len(v13), f"Length mismatch: v13.2={len(v132)}, v13={len(v13)}"
assert (v132['id'] == v13['id']).all(), "ID order mismatch"

is_val  = v132['id'].str.contains('_validation')
is_eval = v132['id'].str.contains('_evaluation')
print(f"Validation rows: {is_val.sum():,}")
print(f"Evaluation rows: {is_eval.sum():,}")

print(f"\nTotals before blending:")
print(f"  v13.2 val:  {v132.loc[is_val, F_cols].values.sum():,.0f}")
print(f"  v13.2 eval: {v132.loc[is_eval, F_cols].values.sum():,.0f}")
print(f"  v13   val:  {v13.loc[is_val, F_cols].values.sum():,.0f}")
print(f"  v13   eval: {v13.loc[is_eval, F_cols].values.sum():,.0f}")

ALPHAS = {
    'A_90_10': 0.90,
    'B_80_20': 0.80,
    'C_70_30': 0.70,
}

for label, alpha in ALPHAS.items():
    blend = v132.copy()
    eval_v132 = v132.loc[is_eval, F_cols].values
    eval_v13  = v13.loc[is_eval, F_cols].values
    blended_eval = alpha * eval_v132 + (1 - alpha) * eval_v13
    blend.loc[is_eval, F_cols] = blended_eval.astype(np.float32)

    val_diff = np.abs(blend.loc[is_val, F_cols].values - v132.loc[is_val, F_cols].values).sum()
    assert val_diff < 1e-6, f"Validation got modified! diff={val_diff}"

    eval_sum_new = blend.loc[is_eval, F_cols].values.sum()
    eval_sum_v132 = v132.loc[is_eval, F_cols].values.sum()
    eval_sum_v13  = v13.loc[is_eval, F_cols].values.sum()

    out_path = OUT / f"submission_v13_4_{label}.csv"
    blend.to_csv(out_path, index=False)

    print(f"\n=== Blend {label} (alpha={alpha}) ===")
    print(f"  Eval total: {eval_sum_v132:,.0f} (v13.2) → {eval_sum_new:,.0f} (blend) → {eval_sum_v13:,.0f} (v13)")
    print(f"  Eval reduction from v13.2: {(eval_sum_v132 - eval_sum_new)/eval_sum_v132*100:.2f}%")
    print(f"  Saved: {out_path}")

print("\n✅ Done. Submit one of the 3 blends.")
print("  - 90/10 = mild hedge")
print("  - 80/20 = moderate hedge")
print("  - 70/30 = strong hedge")
print("\nALL THREE have IDENTICAL Public LB = 0.48679 (validation unchanged).")
print("Choose based on how much you trust v13.2's eval calibration.")
