# Model Weights

Trained model checkpoints for both test submissions.

---

## Files

| File | Task | Macro AUC | Val Acc | Notes |
|---|---|---|---|---|
| `best_model.pth` | Common Test I | **0.9553** | 84.84% | EfficientNet-B3, 30 epochs |
| `best_pinn_v2.pth` | Test VII (PINN) | **0.9578** | 84.64% | LensPINN-v2, 40 epochs |

---


## Loading

### Common Test I baseline

```python
import torch
from torchvision import models

model = models.efficientnet_b3(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 3)
model.load_state_dict(torch.load('models/best_baseline.pth', map_location='cpu'))
model.eval()
```

### LensPINN-v2

```python
import torch
import sys
sys.path.insert(0, '.')
from test7_pinn.final_model.model_architecture import build_model

model = build_model(num_classes=3, img_size=224)
model.load_state_dict(torch.load('models/best_pinn_v2.pth', map_location='cpu'))
model.eval()
```

---

## Training Environment

- Platform: Kaggle (NVIDIA T4 × 2, 16 GB VRAM each)
- CUDA: 11.8
- PyTorch: 2.0.x
- Training time: ~45 min (baseline) / ~2.5 hrs (LensPINN-v2)
