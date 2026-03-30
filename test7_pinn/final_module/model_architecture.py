"""
LensPINN-v2: Physics-Informed EfficientNet for Gravitational Lens Classification
=================================================================================
GSoC 2026 — ML4Sci DEEPLENSE | Specific Test VII
Author: Parv Patel | IIT Palakkad

Architecture overview:
    Input Image
        │
        ├──[ConvergenceNet (U-Net)]──► κ̂ map  [Softplus, κ ≥ 0]
        │                                  │
        │                        [PhysicsLayer (no params)]
        │                        ∇²ψ = 2κ  → FFT Poisson solve
        │                        α  = ∇ψ   → spectral differentiation
        │                        β  = θ−α  → source reconstruction (grid_sample)
        │                        outputs: [κ, γ₁, γ₂, |α|, μ]  (5 maps)
        │                                  │
        │                    [PhysicsFeatureEncoder]──► phys_feat (256-d)
        │                                                    │
        └──[EfficientNet-B3 (pretrained)]──────────────────►│
                                                   spatial_feat (1536-d)
                                                         │
                                                  Gated Fusion (1792-d)
                                                         │
                                                     MLP Head
                                                         │
                                                   3-class logits

References:
    LensPINN  — Ojha et al., NeurIPS ML4PS 2024
    Lensiformer — Velôso et al., NeurIPS ML4PS 2023
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ──────────────────────────────────────────────────────────────────────────────
# 1. PhysicsLayer — differentiable gravitational lensing equations
# ──────────────────────────────────────────────────────────────────────────────

class PhysicsLayer(nn.Module):
    """
    Differentiable implementation of gravitational lensing equations.
    Contains zero learnable parameters — entirely analytical.
    Gradients propagate through torch.fft and F.grid_sample back into
    ConvergenceNet at every training step.

    Physics pipeline:
        κ  →  ∇²ψ = 2κ  (Poisson, solved via 2-D FFT)
           →  α = ∇ψ    (deflection field, spectral differentiation)
           →  β = θ − α  (source plane positions, lens equation)
           →  γ₁, γ₂, μ  (shear components, magnification)

    Output maps:
        Channel 0: κ         convergence (mass density proxy)
        Channel 1: γ₁        horizontal/vertical shear
        Channel 2: γ₂        diagonal shear  ← key for vortex detection
        Channel 3: |α|       deflection magnitude
        Channel 4: μ         magnification (clipped for stability)
    """

    def __init__(self, img_size: int = 224):
        super().__init__()
        H = W = img_size

        # Frequency grids for Poisson solver — not learned, registered as buffers
        ky = torch.fft.fftfreq(H).view(H, 1).expand(H, W) * 2 * np.pi
        kx = torch.fft.fftfreq(W).view(1, W).expand(H, W) * 2 * np.pi
        k2 = kx ** 2 + ky ** 2
        k2[0, 0] = 1.0  # avoid DC division by zero; DC of ψ is arbitrary

        self.register_buffer('KX', kx)
        self.register_buffer('KY', ky)
        self.register_buffer('K2', k2)

        # Image-plane coordinate grid θ ∈ [-1,1]² for grid_sample
        grid_y = torch.linspace(-1, 1, H).view(H, 1).expand(H, W)
        grid_x = torch.linspace(-1, 1, W).view(1, W).expand(H, W)
        self.register_buffer('GRID_X', grid_x)
        self.register_buffer('GRID_Y', grid_y)

    def forward(self, kappa: torch.Tensor, image_raw: torch.Tensor):
        """
        Args:
            kappa     : (B, 1, H, W) — predicted convergence map, κ ≥ 0
            image_raw : (B, 3, H, W) — raw [0,1] image (not ImageNet-normalised)

        Returns:
            physics_maps : (B, 5, H, W) — [κ, γ₁, γ₂, |α|, μ]
            source_recon : (B, 3, H, W) — source-plane reconstruction via β = θ − α
        """
        B = kappa.shape[0]
        kappa_sq = kappa.squeeze(1)  # (B, H, W)

        # ── Poisson solve: ∇²ψ = 2κ in Fourier space ────────────────────────
        kappa_hat = torch.fft.fft2(2.0 * kappa_sq)          # (B, H, W) complex
        psi_hat   = kappa_hat / self.K2.unsqueeze(0)         # ψ̂ = κ̂ / k²

        # ── Deflection angles: α = ∇ψ (spectral differentiation) ────────────
        alpha_x = torch.fft.ifft2(1j * self.KX.unsqueeze(0) * psi_hat).real
        alpha_y = torch.fft.ifft2(1j * self.KY.unsqueeze(0) * psi_hat).real

        # Normalise to prevent grid_sample sampling outside [-1,1]
        scale_x = alpha_x.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        scale_y = alpha_y.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        alpha_x = (alpha_x / scale_x) * 0.3
        alpha_y = (alpha_y / scale_y) * 0.3

        # ── Lens equation: β = θ − α ─────────────────────────────────────────
        theta_x = self.GRID_X.unsqueeze(0).expand(B, -1, -1)
        theta_y = self.GRID_Y.unsqueeze(0).expand(B, -1, -1)
        beta_x  = (theta_x - alpha_x).clamp(-1, 1)
        beta_y  = (theta_y - alpha_y).clamp(-1, 1)
        beta    = torch.stack([beta_x, beta_y], dim=-1)  # (B, H, W, 2)

        source_recon = F.grid_sample(
            image_raw, beta,
            mode='bilinear', padding_mode='reflection', align_corners=True
        )  # (B, 3, H, W)

        # ── Shear and magnification from second derivatives of ψ ─────────────
        psi_xx = torch.fft.ifft2(-self.KX.unsqueeze(0) ** 2 * psi_hat).real
        psi_yy = torch.fft.ifft2(-self.KY.unsqueeze(0) ** 2 * psi_hat).real
        psi_xy = torch.fft.ifft2(-self.KX.unsqueeze(0) * self.KY.unsqueeze(0) * psi_hat).real

        kappa_recon = (psi_xx + psi_yy) / 2.0
        gamma1      = (psi_xx - psi_yy) / 2.0
        gamma2      = psi_xy
        gamma_sq    = gamma1 ** 2 + gamma2 ** 2

        # Magnification — clipped for stability near critical curves
        denom = (1 - kappa_recon) ** 2 - gamma_sq
        mu    = (1.0 / (
            denom.abs().clamp(min=1e-3) * denom.sign().clamp(min=0) + 1e-3
        )).clamp(-10, 10)

        alpha_mag = (alpha_x ** 2 + alpha_y ** 2).sqrt()

        physics_maps = torch.stack([
            kappa_recon, gamma1, gamma2, alpha_mag, mu
        ], dim=1)  # (B, 5, H, W)

        return physics_maps, source_recon


# ──────────────────────────────────────────────────────────────────────────────
# 2. ConvergenceNet — U-Net predicting κ̂(x, y)
# ──────────────────────────────────────────────────────────────────────────────

class ConvergenceNet(nn.Module):
    """
    U-Net encoder-decoder predicting the convergence map κ̂(x, y).

    Design choices:
        - Softplus output enforces κ ≥ 0 (physical constraint)
        - Skip connections preserve spatial detail needed for localised subhalo peaks
        - BatchNorm + GELU throughout for stable training
        - Lightweight (~2M params) to avoid dominating the EfficientNet branch
    """

    def __init__(self):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.GELU(),
            )

        # Encoder
        self.enc1 = block(3, 32)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), block(32, 64))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), block(64, 128))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2), block(128, 256))

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            block(256, 512),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            block(512, 256),
        )

        # Decoder (with skip connections)
        self.dec4 = nn.Sequential(
            block(512, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.dec3 = nn.Sequential(
            block(256, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.dec2 = nn.Sequential(
            block(128, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.dec1 = block(64, 16)

        # Output head: 1-channel κ map with Softplus to enforce κ ≥ 0
        self.head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 3, H, W) — ImageNet-normalised lensing image

        Returns:
            (B, 1, H, W) — convergence map κ̂, all values ≥ 0
        """
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b  = self.bottleneck(e4)
        d4 = self.dec4(torch.cat([b,  e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        return self.head(d1)  # (B, 1, H, W)


# ──────────────────────────────────────────────────────────────────────────────
# 3. PhysicsFeatureEncoder — encode 5-channel physics maps
# ──────────────────────────────────────────────────────────────────────────────

class PhysicsFeatureEncoder(nn.Module):
    """
    Lightweight CNN that encodes the 5-channel physics maps
    [κ, γ₁, γ₂, |α|, μ] into a fixed-size feature vector.

    γ₂ is particularly important here — it captures rotational shear asymmetry,
    which is the primary distinguishing feature between vortex and sphere classes.
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 16, out_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 5, H, W) — five physics maps

        Returns:
            (B, out_dim) — physics feature vector
        """
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# 4. LensPINN — full model combining all components
# ──────────────────────────────────────────────────────────────────────────────

class LensPINN(nn.Module):
    """
    Physics-Informed Neural Network for gravitational lens classification.

    Two complementary feature streams:
        1. Spatial stream:  EfficientNet-B3 on the observed lensed image  (1536-d)
        2. Physics stream:  ConvergenceNet → PhysicsLayer → PhysicsFeatureEncoder (256-d)

    The physics stream provides features that a purely spatial backbone cannot compute:
    the convergence map κ encodes where mass is concentrated, γ₁/γ₂ encode how that
    mass shears the background galaxy, and μ encodes total flux amplification.
    These quantities differ across dark matter classes in predictable, physically
    interpretable ways.

    Fusion: learned sigmoid gate on the concatenated 1792-d vector, then MLP head.
    """

    def __init__(
        self,
        num_classes: int = 3,
        img_size: int = 224,
        phys_dim: int = 256,
    ):
        super().__init__()

        # ── Physics branch ────────────────────────────────────────────────────
        self.convergence_net = ConvergenceNet()
        self.physics_layer   = PhysicsLayer(img_size)
        self.phys_encoder    = PhysicsFeatureEncoder(out_dim=phys_dim)

        # ── Spatial branch: EfficientNet-B3 ──────────────────────────────────
        eff = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        # Freeze stem — preserves low-level ImageNet edge detectors
        for name, param in eff.named_parameters():
            if 'features.0' in name or 'features.1' in name:
                param.requires_grad = False
        spatial_dim = eff.classifier[1].in_features  # 1536 for B3
        eff.classifier = nn.Identity()
        self.spatial_enc  = eff
        self.spatial_drop = nn.Dropout(0.4)

        # ── Fusion ────────────────────────────────────────────────────────────
        fused_dim = spatial_dim + phys_dim  # 1792

        # Learned sigmoid gate: weight each dimension by its relevance
        self.gate = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.Sigmoid(),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, img_norm: torch.Tensor, img_raw: torch.Tensor):
        """
        Args:
            img_norm : (B, 3, H, W) — ImageNet-normalised, fed to EfficientNet + ConvergenceNet
            img_raw  : (B, 3, H, W) — raw [0,1] pixels, fed to PhysicsLayer for source sampling

        Returns:
            logits       : (B, num_classes)
            source_recon : (B, 3, H, W) — physics-reconstructed source galaxy
            kappa        : (B, 1, H, W) — predicted convergence map
        """
        # Physics branch
        kappa                    = self.convergence_net(img_raw)
        phys_maps, source_recon  = self.physics_layer(kappa, img_raw)
        phys_feat                = self.phys_encoder(phys_maps)

        # Spatial branch
        spatial_feat = self.spatial_drop(self.spatial_enc(img_norm))

        # Gated fusion
        fused  = torch.cat([spatial_feat, phys_feat], dim=1)
        gated  = self.gate(fused) * fused
        logits = self.classifier(gated)

        return logits, source_recon, kappa


# ──────────────────────────────────────────────────────────────────────────────
# Convenience function
# ──────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = 3, img_size: int = 224) -> LensPINN:
    """Instantiate LensPINN-v2 with default hyperparameters."""
    return LensPINN(num_classes=num_classes, img_size=img_size, phys_dim=256)


def count_parameters(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total":     total,
        "trainable": trainable,
        "frozen":    total - trainable,
        "pct_trainable": f"{100 * trainable / total:.1f}%",
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model().to(device)
    stats  = count_parameters(model)
    print(f"LensPINN-v2 parameters:")
    for k, v in stats.items():
        print(f"  {k:>16}: {v:,}" if isinstance(v, int) else f"  {k:>16}: {v}")

    # Smoke test
    img_n = torch.randn(2, 3, 224, 224).to(device)
    img_r = torch.rand(2,  3, 224, 224).to(device)
    with torch.no_grad():
        logits, src, kappa = model(img_n, img_r)
    print(f"\nForward pass shapes:")
    print(f"  logits       : {logits.shape}")
    print(f"  source_recon : {src.shape}")
    print(f"  kappa        : {kappa.shape}")
    print(f"  kappa range  : [{kappa.min():.4f}, {kappa.max():.4f}]  (should be ≥ 0)")