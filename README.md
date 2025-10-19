# TGN Gen 2.5 (CIFAR-100)

Sharpened hierarchical classifier for CIFAR-100: SAM + EMA tuned, MixUp ready, dashboard included.

![](https://dummyimage.com/900x320/111827/ffffff&text=Taxonomic+Gated+Network+Gen+2.5)

---

## What's Inside

| File | Description |
|------|-------------|
| `tgn2_5.py` | Taxonomic Gated Network (ResNet-18 trunk, 20 experts) |
| `train_tgn2_5.py` | Training loop, CLI, live-dashboard hooks |
| `tgn_gen2_5_best.pt` | Best checkpoint (epoch 101) |
| `visualize_tgn.py` | Layer visualizer and feature explorer |

**Total Parameters:** 16,609,720

---

## Benchmark Snapshot

| Epoch | Train Fine | Train Super | Val Fine | Val Super | Loss | Time / Epoch |
|-------|-----------:|------------:|---------:|----------:|-----:|-------------:|
| 101 | 98.46 % | 98.46 % | **73.04 %** | **82.64 %** | 0.690 | ~97 s (RTX 4060, SAM+EMA) |

These numbers were obtained with SAM (`rho=0.05`), EMA (`0.9999`), MixUp (alpha 0.2), label smoothing (0.1), channels-last tensors, and persistent dataloader workers.

---


### Git LFS requirement
This repository stores the checkpoint with [Git LFS](https://git-lfs.com/). Install it first:
```bash
git lfs install
git lfs pull
```
## Quick Start

```bash
# Install dependencies (example with uv)
uv pip install torch torchvision torch-ema

# Kick off training (150 epochs, resume if checkpoint exists)
uv run python train_tgn2_5.py --epochs 150 --use-sam --warmup-epochs 10 --resume auto
```

Live metrics stream to http://localhost:8891/dashboard.html.

---

## Architecture Highlights

- ResNet-18 trunk (stem adapted for 32x32 inputs)
- 20 supervised experts (one per CIFAR-100 superclass)
- Gate learns superclass logits, experts produce subclass logits
- MixUp-aware hierarchical loss with label smoothing
- Channels-last + AMP by default for modern GPUs

> Peek inside the experts: use `visualize_tgn.py` (included) to inspect layer activations and attention heatmaps.

---

## CLI Cheatsheet

```bash
uv run python train_tgn2_5.py --help

uv run python train_tgn2_5.py \
  --epochs 50 --use-sam --warmup-epochs 5 \
  --mixup-alpha 0.2 --grad-accum 1

# Resume the included checkpoint
uv run python train_tgn2_5.py --resume TGN-cifar-100/tgn_gen2_5_best.pt
```

Checkpoints are stored in `checkpoints_tgn_gen2_5/` by default. SAM/EMA stay enabled unless you add `--no-ema` or skip `--use-sam`.

---

## Notes

- CIFAR-100 dataset files are not bundled; place them in `./cifar100/` or let torchvision download automatically.
- Tested on PyTorch 2.4 (CUDA 12.4 wheels). Adjust if you are on another toolkit.
- The dashboard server runs in-process; stop the script to close it.

---

## Contribute / Extend

This release is a clean baseline. Fork it, swap in ConvNeXt or Swin trunks, add prototypes, or experiment with contrastive objectives—and let us know how far you push Gen 2.5.

Happy training!
