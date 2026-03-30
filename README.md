# DeepLense_LensPINN

**GSoC 2026 Application — ML4Sci DEEPLENSE**  
**Task:** Common Test I (Multi-Class Classification) + Specific Test VII (Physics-Guided ML)  
**Author:** Parv Patel | IIT Palakkad  

---

## Results at a Glance

| Model | No Sub AUC | Sphere AUC | Vortex AUC | Macro AUC |
|---|---|---|---|---|
| EfficientNet-B3 baseline (Test I) | 0.9695 | 0.9325 | 0.9635 | **0.9553** |
| LensPINN v1 (naive training) | 0.9648 | 0.9202 | 0.9502 | 0.9452 |
| **LensPINN v2 (this work)** | **0.9747** | **0.9389** | **0.9594** | **0.9578** |

All results are reported on the held-out `val/` split (7,500 images, 2,500 per class). TTA with 8 augmented views (D4 symmetry group).

---

## Repository Structure

```
DeepLense-LensPINN-v2/
│
├── README.md
│
├── test1_baseline/          ← Common Test I
│   ├── notebook.ipynb       ← EfficientNet-B3 classifier (AUC 0.9553)
│   ├── config.yaml          ← all hyperparameters
│   └── results/             ← ROC curve, training history, sample images
│
├── test7_pinn/              ← Specific Test VII
│   ├── initial_attempt/     ← LensPINN v1 (AUC 0.9452, underperforms baseline)
│   └── final_model/         ← LensPINN v2 (AUC 0.9578, beats baseline)
│       ├── notebook.ipynb
│       ├── model_architecture.py
│       └── results/
│
├── models/                  ← trained weight files (see models/README.md)
│   ├── best_model.pth
│   ├── best_pinn_v2.pth
│   └── README.md
│
├── results_summary/         ← cross-experiment comparison figures
│   ├── roc_comparison.png
│   ├── training_curves.png
│   └── README.md
│
│
└── requirements.txt
```

---

## Task Descriptions

### Common Test I — Multi-Class Classification

Classify simulated gravitational lensing images into three dark matter substructure classes using EfficientNet-B3 with transfer learning.

**Key design decisions:**

| Decision | Rationale |
|---|---|
| EfficientNet-B3 | Best accuracy/compute trade-off for fine-grained spatial classification |
| Rotational augmentation (0/90/180/270°) | Lensing images are rotationally symmetric — physically motivated |
| Label smoothing ε=0.1 | Substructure class boundaries are soft |
| WeightedRandomSampler | Balanced training without discarding samples |
| Differential LR (backbone 10× lower) | Preserves ImageNet features; prevents catastrophic forgetting |
| TTA (8-view D4 symmetry) | Further AUC improvement at inference, zero training cost |

### Specific Test VII — Physics-Guided ML (LensPINN-v2)

Embed the gravitational lens equation into the architecture to extract physically meaningful features alongside spatial CNN features.

**Architecture:**
```
Input Image
    │
    ├──[ConvergenceNet (U-Net)]──► κ̂ map (B,1,H,W)  [κ ≥ 0 via Softplus]
    │                                    │
    │                          [PhysicsLayer (no params)]
    │                          ∇²ψ = 2κ  → FFT Poisson solve
    │                          α = ∇ψ    → deflection angles
    │                          β = θ − α → source reconstruction
    │                          γ₁,γ₂,μ  → shear, magnification
    │                                    │
    │                       Physics maps (B,5,H,W)
    │                                    │
    │                      [PhysicsFeatureEncoder]──► phys_feat (256-d)
    │                                                      │
    └──[EfficientNet-B3]──────────────────────────────────►│
                                                    spatial_feat (1536-d)
                                                           │
                                                    Gated Fusion (1792-d)
                                                           │
                                                       MLP Head
                                                           │
                                                     Class Logits
```

**Why LensPINN v1 failed and how v2 fixes it:**

| Issue | v1 (broken) | v2 (fixed) |
|---|---|---|
| λ_phys fixed from epoch 1 | Physics loss dominated before ConvergenceNet learned anything | **λ warmup**: 0 → 0.15 over first 5 epochs |
| ConvergenceNet gradients live from epoch 1 | Noisy κ maps destabilised the classifier | **Gradient detach** for epochs 1–3 |
| Same λ for all classes | Penalised subhalo's naturally irregular κ | **Class-conditional λ**: sphere gets 0.5× |
| Early stopping on total loss | Physics loss variance triggered false stops | Early stop on **classification loss only** |

---

## Dataset Structure

```
dataset/
  train/
    no/       → class 0 (no substructure)
    sphere/   → class 1 (CDM subhalo)
    vort/     → class 2 (axion vortex)
  val/
    no/ | sphere/ | vort/    ← final evaluation (never touched during training)
```

10,000 training images per class (30,000 total), 2,500 val images per class.  
Files are single-channel `.npy` arrays, min-max normalised to [0,1].

---

## Quick Start

```bash
git clone https://github.com/Parvptl/DeepLense_LensPINN
cd DeepLense_LensPINN
pip install -r requirements.txt
```

Update `data_dir` in `test1_baseline/config.yaml` or the CONFIG cell of each notebook to point to your local dataset path, then run the notebooks top-to-bottom.

---

## References

- Ojha et al. (2024) — *LensPINN: Physics Informed Neural Network for Learning Dark Matter Morphology in Lensing*. NeurIPS ML4PS Workshop.
- Srivastava et al. (2025) — *HEAL-PINN: Physics-Informed Swin Transformer for Dark Matter Studies*. NeurIPS ML4PS Workshop.
- Vel\^oso et al. (2023) — *Lensformer: A Physics-Informed Vision Transformer for Gravitational Lensing*. NeurIPS ML4PS Workshop.
- Alexander et al. (2020) — *Deep Learning the Morphology of Dark Matter Substructure*. ApJ 893(1):15.
