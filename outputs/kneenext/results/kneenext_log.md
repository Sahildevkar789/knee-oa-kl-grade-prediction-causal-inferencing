# KneeNeXt ablation run started

Device: cuda | Train: 5778 | Val: 826

Joint-line detector: check figure saved to `outputs/kneenext/results\kneenext_joint_zoom_check.png`. Failure/fallback rate over val+train: 0.0002 (0.02%) - falls back to the vertical center when the 'darkest row' isn't a meaningful dip (z-score < 1.0).

- `kneenext_control_seed42` succeeded on attempt 1 (batch size 32): val QWK 0.8381, val acc 0.6804, 14.2 min, 27,897,030 params

- `kneenext_control_seed7` succeeded on attempt 1 (batch size 32): val QWK 0.8399, val acc 0.6683, 13.4 min, 27,897,030 params

## Config `control`: mean val QWK 0.8390 (seeds: [0.8381, 0.8399]) vs best-so-far N/A (first config) -> **ADOPTED**

- `kneenext_joint_zoom_seed42` succeeded on attempt 1 (batch size 32): val QWK 0.8375, val acc 0.6913, 19.0 min, 27,901,638 params

- `kneenext_joint_zoom_seed7` succeeded on attempt 1 (batch size 32): val QWK 0.8399, val acc 0.6695, 19.1 min, 27,901,638 params

## Config `joint_zoom`: mean val QWK 0.8387 (seeds: [0.8375, 0.8399]) vs best-so-far 0.839009268732714 -> **NOT ADOPTED**

- `kneenext_multiscale_seed42` succeeded on attempt 1 (batch size 32): val QWK 0.8364, val acc 0.6598, 19.1 min, 28,197,318 params

- `kneenext_multiscale_seed7` succeeded on attempt 1 (batch size 32): val QWK 0.8360, val acc 0.6586, 17.1 min, 28,197,318 params

## Config `multiscale`: mean val QWK 0.8362 (seeds: [0.8364, 0.836]) vs best-so-far 0.839009268732714 -> **NOT ADOPTED**

- `kneenext_joint_cbam_seed42` succeeded on attempt 1 (batch size 32): val QWK 0.8385, val acc 0.6622, 29.1 min, 28,197,319 params, alpha=1.175

- `kneenext_joint_cbam_seed7` succeeded on attempt 1 (batch size 32): val QWK 0.8380, val acc 0.6792, 17.8 min, 28,197,319 params, alpha=1.173

## Config `joint_cbam`: mean val QWK 0.8383 (seeds: [0.8385, 0.838]) vs best-so-far 0.839009268732714 -> **NOT ADOPTED**

- `kneenext_asymmetry_seed42` succeeded on attempt 1 (batch size 32): val QWK 0.8448, val acc 0.6695, 17.7 min, 28,201,927 params, alpha=1.168

- `kneenext_asymmetry_seed7` succeeded on attempt 1 (batch size 32): val QWK 0.8443, val acc 0.6755, 19.6 min, 28,201,927 params, alpha=1.153

## Config `asymmetry`: mean val QWK 0.8446 (seeds: [0.8448, 0.8443]) vs best-so-far 0.839009268732714 -> **ADOPTED**

Ablation table saved to `outputs/kneenext/results\kneenext_ablation.csv`.

**Overall best KneeNeXt config: `asymmetry` (seed 42, val QWK 0.8448)** - copied to `outputs/kneenext/checkpoints\kneenext_best.pt`, flags saved to `kneenext_best_config.json`.

# KneeNeXt ablation run finished.

