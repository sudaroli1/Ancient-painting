"""
Training script for the Coordinated Attention cGAN with Diffusion Discriminator.

Usage:
    python train.py --data_root /path/to/murals --epochs 100 --batch_size 8
"""

import argparse
import os
import time
import logging
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from model          import CoordAttentionGenerator, DiffusionDiscriminator, \
                           DiffusionDiscriminatorLoss, InpaintingLoss
from dataset        import MuralDataset
from metrics        import MetricTracker, psnr, ssim

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────
#  Argument parser
# ─────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Mural restoration training")
    p.add_argument("--data_root",   type=str, required=True)
    p.add_argument("--output_dir",  type=str, default="./checkpoints")
    p.add_argument("--image_size",  type=int, default=256)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--epochs",      type=int, default=100)
    p.add_argument("--lr_g",        type=float, default=2e-4)
    p.add_argument("--lr_d",        type=float, default=2e-4)
    p.add_argument("--base_ch",     type=int, default=64)
    p.add_argument("--save_every",  type=int, default=10,
                   help="Save checkpoint every N epochs")
    p.add_argument("--val_every",   type=int, default=5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume",      type=str, default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


# ─────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────

def train_one_epoch(G, D, opt_G, opt_D,
                    loader, loss_fn_D, loss_fn_G,
                    device, tracker):
    G.train(); D.train()
    for masked, mask, gt in loader:
        masked, mask, gt = masked.to(device), mask.to(device), gt.to(device)

        # ── Train Discriminator ───────────────
        opt_D.zero_grad()
        fake      = G(masked, mask).detach()
        d_loss, l_de, l_dd = loss_fn_D.discriminator_loss(D, gt, fake)
        d_loss.backward()
        opt_D.step()

        # ── Train Generator ───────────────────
        opt_G.zero_grad()
        fake     = G(masked, mask)
        adv_loss = loss_fn_D.generator_loss(D, fake)
        g_loss   = loss_fn_G(fake, gt, mask, adv_loss)
        g_loss.backward()
        opt_G.step()

        tracker.update(
            d_loss  = d_loss.item(),
            g_loss  = g_loss.item(),
            L_DE    = l_de,
            L_Dd    = l_dd,
            psnr_batch = psnr(fake.detach(), gt),
            ssim_batch = ssim(fake.detach(), gt),
        )


@torch.no_grad()
def validate(G, loader, device, tracker, output_dir, epoch):
    G.eval()
    sample_saved = False
    for masked, mask, gt in loader:
        masked, mask, gt = masked.to(device), mask.to(device), gt.to(device)
        fake = G(masked, mask)
        tracker.update(
            val_psnr = psnr(fake, gt),
            val_ssim = ssim(fake, gt),
        )
        if not sample_saved:
            grid = torch.cat([masked[:4], fake[:4], gt[:4]], dim=0)
            save_image(grid * 0.5 + 0.5,
                       os.path.join(output_dir, f"val_epoch{epoch:04d}.png"),
                       nrow=4)
            sample_saved = True


# ─────────────────────────────────────────
#  Main
# ─────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Data
    train_ds = MuralDataset(args.data_root, split="train",
                             image_size=args.image_size)
    val_ds   = MuralDataset(args.data_root, split="val",
                             image_size=args.image_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True,  num_workers=args.num_workers,
                               pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers,
                               pin_memory=True)

    # Models
    G = CoordAttentionGenerator(base_ch=args.base_ch).to(device)
    D = DiffusionDiscriminator(base_ch=args.base_ch).to(device)

    # Optimisers
    opt_G = optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    # LR schedulers (cosine annealing for smooth decay)
    sched_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=args.epochs)
    sched_D = optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=args.epochs)

    # Loss functions
    loss_fn_D = DiffusionDiscriminatorLoss().to(device)
    loss_fn_G = InpaintingLoss().to(device)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        start_epoch = ckpt["epoch"] + 1
        log.info(f"Resumed from epoch {start_epoch}")

    tracker = MetricTracker()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tracker.reset()

        train_one_epoch(G, D, opt_G, opt_D,
                        train_loader, loss_fn_D, loss_fn_G,
                        device, tracker)
        sched_G.step(); sched_D.step()

        elapsed = time.time() - t0
        report  = tracker.report()
        log.info(
            f"Epoch {epoch+1:04d}/{args.epochs} | "
            f"G={report.get('g_loss',0):.4f} D={report.get('d_loss',0):.4f} "
            f"PSNR={report.get('psnr_batch',0):.2f} dB "
            f"SSIM={report.get('ssim_batch',0):.4f} | {elapsed:.1f}s"
        )

        if (epoch + 1) % args.val_every == 0:
            validate(G, val_loader, device, tracker,
                     args.output_dir, epoch + 1)
            vr = tracker.report()
            log.info(f"  → val PSNR={vr.get('val_psnr',0):.2f} dB "
                     f"val SSIM={vr.get('val_ssim',0):.4f}")

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir,
                                     f"ckpt_epoch{epoch+1:04d}.pth")
            torch.save({
                "epoch": epoch,
                "G": G.state_dict(),
                "D": D.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
            }, ckpt_path)
            log.info(f"  → Saved checkpoint: {ckpt_path}")

    # Final save
    torch.save(G.state_dict(),
               os.path.join(args.output_dir, "generator_final.pth"))
    log.info("Training complete.")


if __name__ == "__main__":
    main()
