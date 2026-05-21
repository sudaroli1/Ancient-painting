"""
Coordinated Attention Conditional GAN with Diffusion Discriminator
for Historic Mural Image Restoration

Paper: Physical restoration in conjunction with digital restoration,
employing a coordinated conditional GAN with diffusion discriminator on Mural images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────
#  1. COORDINATED ATTENTION BLOCK
# ─────────────────────────────────────────────

class CoordinatedAttention(nn.Module):
    """
    Implements Eq. (1)-(4) from the paper.
    Decomposes global average pooling into separate horizontal and vertical
    1-D encodings, then fuses them to produce spatial-aware attention maps.
    """
    def __init__(self, in_channels: int, reduction: int = 32):
        super().__init__()
        mid = max(8, in_channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))   # (B,C,H,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))   # (B,C,1,W)

        self.conv1   = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.bn1     = nn.BatchNorm2d(mid)
        self.act     = nn.Hardswish()

        self.conv_h  = nn.Conv2d(mid, in_channels, 1, bias=False)
        self.conv_w  = nn.Conv2d(mid, in_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Eq.(2): dec_h  shape (B,C,H,1)
        dec_h = self.pool_h(x)
        # Eq.(3): dec_w  shape (B,C,1,W) → transpose to (B,C,W,1)
        dec_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Eq.(4): fuse along the spatial dimension, apply non-linear transform f
        fused = torch.cat([dec_h, dec_w], dim=2)          # (B,C,H+W,1)
        fused = self.act(self.bn1(self.conv1(fused)))      # (B,mid,H+W,1)

        # Split back into horizontal / vertical halves
        feat_h, feat_w = fused.split([H, W], dim=2)

        # Attention weights (no sigmoid; paper removes it to improve coherence)
        attn_h = self.conv_h(feat_h)                       # (B,C,H,1)
        attn_w = self.conv_w(feat_w.permute(0, 1, 3, 2))  # (B,C,1,W)

        return x * attn_h * attn_w


# ─────────────────────────────────────────────
#  2. RESIDUAL BLOCK WITH COORDINATED ATTENTION
# ─────────────────────────────────────────────

class DilatedResBlock(nn.Module):
    """
    Dilated convolution + CoordinatedAttention + residual shortcut.
    Multiple dilation rates are fused to capture multi-scale context.
    """
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.attn  = CoordinatedAttention(channels)
        self.gate  = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attn(out)
        out = out * self.gate(out)          # gated attention (Fig. 2b)
        return F.relu(out + x)             # residual shortcut


# ─────────────────────────────────────────────
#  3. GENERATOR
# ─────────────────────────────────────────────

class CoordAttentionGenerator(nn.Module):
    """
    Encoder-bottleneck-decoder generator with:
      - Multi-scale dilated residual blocks (dilations 1,2,4,8)
      - CoordinatedAttention fused into every residual block
      - Skip connections from encoder to decoder
    Input: masked image (3 ch) + binary mask (1 ch) = 4 channels
    Output: completed image (3 ch)
    """
    def __init__(self, base_ch: int = 64):
        super().__init__()
        # ── Encoder ──────────────────────────────
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, base_ch, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2)
        )
        self.enc2 = self._enc_block(base_ch,     base_ch * 2)
        self.enc3 = self._enc_block(base_ch * 2, base_ch * 4)
        self.enc4 = self._enc_block(base_ch * 4, base_ch * 8)

        # ── Bottleneck with multi-dilation blocks ─
        self.bottleneck = nn.Sequential(
            DilatedResBlock(base_ch * 8, dilation=1),
            DilatedResBlock(base_ch * 8, dilation=2),
            DilatedResBlock(base_ch * 8, dilation=4),
            DilatedResBlock(base_ch * 8, dilation=8),
        )

        # ── Decoder with skip connections ─────────
        self.dec4 = self._dec_block(base_ch * 8 * 2, base_ch * 4)
        self.dec3 = self._dec_block(base_ch * 4 * 2, base_ch * 2)
        self.dec2 = self._dec_block(base_ch * 2 * 2, base_ch)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, 3, 4, 2, 1),
            nn.Tanh()
        )

    @staticmethod
    def _enc_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        )

    @staticmethod
    def _dec_block(in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, mask], dim=1)   # 4-channel input
        e1  = self.enc1(inp)
        e2  = self.enc2(e1)
        e3  = self.enc3(e2)
        e4  = self.enc4(e3)
        b   = self.bottleneck(e4)
        d4  = self.dec4(torch.cat([b,  e4], dim=1))
        d3  = self.dec3(torch.cat([d4, e3], dim=1))
        d2  = self.dec2(torch.cat([d3, e2], dim=1))
        out = self.dec1(torch.cat([d2, e1], dim=1))
        # Composite: keep original pixels outside mask
        return x * (1 - mask) + out * mask


# ─────────────────────────────────────────────
#  4. DIFFUSION-BASED DISCRIMINATOR  (Eq. 5-6)
# ─────────────────────────────────────────────

class DiffusionDiscriminator(nn.Module):
    """
    U-Net-style discriminator (encoder + decoder) with skip connections.
    Eq.(5):  L_Ddif = L_DE_dif  +  L_Dd_dif
    Eq.(6):  L_DE   = -E_in[log DE(in)] - E_out[log(1-DE(G(out)))]
    The decoder produces pixel-wise real/fake maps (patch discrimination).
    """
    def __init__(self, in_ch: int = 3, base_ch: int = 64):
        super().__init__()
        # ── Encoder (D_E) ────────────────────────
        self.e1 = self._enc(in_ch,       base_ch,     bn=False)
        self.e2 = self._enc(base_ch,     base_ch * 2)
        self.e3 = self._enc(base_ch * 2, base_ch * 4)
        self.e4 = self._enc(base_ch * 4, base_ch * 8)

        # Scalar output for encoder loss (Eq.6)
        self.enc_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch * 8, 1)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(base_ch * 8),
            nn.LeakyReLU(0.2)
        )

        # ── Decoder (D_d): pixel-wise map ────────
        self.d4 = self._dec(base_ch * 8 * 2, base_ch * 4)
        self.d3 = self._dec(base_ch * 4 * 2, base_ch * 2)
        self.d2 = self._dec(base_ch * 2 * 2, base_ch)
        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, 1, 4, 2, 1),
            nn.Sigmoid()   # per-pixel real/fake score
        )

    @staticmethod
    def _enc(in_ch, out_ch, bn=True):
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not bn)]
        if bn: layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    @staticmethod
    def _dec(in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        enc_score = self.enc_head(e4)       # scalar  → Eq.(6)
        b  = self.bottleneck(e4)

        d4 = self.d4(torch.cat([b,  e4], dim=1))
        d3 = self.d3(torch.cat([d4, e3], dim=1))
        d2 = self.d2(torch.cat([d3, e2], dim=1))
        dec_map = self.d1(torch.cat([d2, e1], dim=1))   # pixel map → Eq.(7)

        return enc_score, dec_map


# ─────────────────────────────────────────────
#  5. LOSS FUNCTIONS
# ─────────────────────────────────────────────

class DiffusionDiscriminatorLoss(nn.Module):
    """Implements Eq.(5) and Eq.(6) from the paper."""

    def __init__(self):
        super().__init__()
        self.bce   = nn.BCEWithLogitsLoss()
        self.bce_s = nn.BCEWithLogitsLoss()

    def discriminator_loss(self, D, real, fake):
        """
        Eq.(6):
          L_DE = -E_in[log DE(in)] - E_out[log(1-DE(G(out)))]
        Combined encoder scalar loss + decoder pixel-map loss = Eq.(5).
        """
        enc_real, dec_real = D(real)
        enc_fake, dec_fake = D(fake.detach())

        # ── Encoder scalar loss (Eq.6) ──────────
        ones  = torch.ones_like(enc_real)
        zeros = torch.zeros_like(enc_fake)
        L_DE  = self.bce_s(enc_real, ones) + self.bce_s(enc_fake, zeros)

        # ── Decoder pixel-map loss (Eq.7 analogy)
        ones_map  = torch.ones_like(dec_real)
        zeros_map = torch.zeros_like(dec_fake)
        L_Dd      = self.bce(dec_real, ones_map) + self.bce(dec_fake, zeros_map)

        # Eq.(5): L_Ddif = L_DE + L_Dd
        return L_DE + L_Dd, L_DE.item(), L_Dd.item()

    def generator_loss(self, D, fake):
        enc_fake, dec_fake = D(fake)
        ones      = torch.ones_like(enc_fake)
        ones_map  = torch.ones_like(dec_fake)
        return self.bce_s(enc_fake, ones) + self.bce(dec_fake, ones_map)


class PerceptualLoss(nn.Module):
    """VGG-16 feature-level loss for structural fidelity."""
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features)[:16]).eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        return F.l1_loss(self.features(pred), self.features(target))


class InpaintingLoss(nn.Module):
    """Combined pixel + perceptual + adversarial loss for the generator."""
    def __init__(self, λ_pixel=1.0, λ_perc=0.1, λ_adv=0.01):
        super().__init__()
        self.λ_pixel = λ_pixel
        self.λ_perc  = λ_perc
        self.λ_adv   = λ_adv
        self.perceptual = PerceptualLoss()

    def forward(self, pred, target, mask, adv_loss):
        # Hole vs. non-hole pixel loss
        hole     = F.l1_loss(pred * mask,       target * mask)
        non_hole = F.l1_loss(pred * (1 - mask), target * (1 - mask))
        pixel    = hole + non_hole

        # Perceptual
        perc = self.perceptual(pred, target)

        return self.λ_pixel * pixel + self.λ_perc * perc + self.λ_adv * adv_loss
