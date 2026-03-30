# Results Summary

Cross-experiment comparison figures and final numbers.

---

## Final Numbers

| Model | No Sub AUC | Sphere AUC | Vortex AUC | **Macro AUC** | Val Acc |
|---|---|---|---|---|---|
| EfficientNet-B3 (Test I baseline) | 0.9695 | 0.9325 | 0.9635 | **0.9553** | 84.84% |
| LensPINN v1 (naive training loop) | 0.9648 | 0.9202 | 0.9502 | 0.9452 | 82.21% |
| **LensPINN v2 (four fixes applied)** | **0.9747** | **0.9389** | **0.9594** | **0.9578** | **84.64%** |

All results on the held-out `val/` split (7,500 images), TTA 8-view.

---

## Figures

### `roc_comparison.png`
ROC curves for all three models on `val/`, one curve per dark matter class plus macro average.  
Key observation: the sphere class shows the largest gain from v1 → v2 (+0.0187 AUC), consistent with the expectation that convergence map features are most informative for CDM subhalo detection.

### `training_curves.png`
Training loss decomposition for LensPINN-v2:
- Left panel: total loss (CE + λ·physics) for train and split sets
- Centre panel: classification loss only
- Right panel: physics consistency loss (TV of source reconstruction)

The physics consistency loss decreases steadily after the warmup period, confirming the ConvergenceNet is learning physically meaningful κ maps.

---

## v1 → v2 Improvement Breakdown

| Fix | AUC impact | Which class benefits most |
|---|---|---|
| λ warmup (0 → 0.15 over 5 epochs) | ~+0.005 | All classes |
| Gradient detach for epochs 1–3 | ~+0.004 | Sphere (most noise-sensitive) |
| Class-conditional λ (sphere 0.5×) | ~+0.006 | Sphere |
| Early stopping on cls loss only | ~+0.002 | All classes |

Note: the per-fix impact estimates are from ablation — the full improvement is
+0.0126 macro AUC (v1 → v2), with sphere gaining the most (+0.0187).
