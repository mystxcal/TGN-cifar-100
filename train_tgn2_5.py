"""
Training script for Taxonomic Gated Network (TGN) - Gen 2.5

Gen 2.5 improvements over baseline ("Free Lunch" stack):
- Expert capacity: 512 hidden units (was 256) - +100% capacity
- MixUp augmentation for regularization (reduces 28% overfitting gap)
- Alpha: 0.85 (was 0.7) - more focus on fine-grained learning
- Label smoothing: 0.1 (softens predictions, better generalization)
- Channels last memory format (+5-10% speed on modern GPUs)
- Persistent workers + prefetch (+10-20% data loading speed)
- Target: 75-80% validation accuracy (Gen 1: 68.9%, Gen 2: 74-78%)

Key features:
- Uses TGN model with ResNet-18 trunk + gating + 20 experts
- Hierarchical loss (superclass + subclass objectives)
- Tracks both superclass and fine-class accuracy
- Separate checkpoints from Gen 1/2 (checkpoints_tgn_gen2_5/)
"""

import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
import time

# Import TGN architecture
from tgn2_5 import TaxonomicGatedNetwork, taxonomic_loss, make_dataloaders, fine_to_coarse, mixup_data, mixup_loss

# Dashboard for live metrics
from dashboard_server import DashboardServer, metrics_store

torch.backends.cudnn.benchmark = True

# Checkpoint directory (Gen 2.5 - separate from Gen 1 & Gen 2)
CHECKPOINT_DIR = Path("checkpoints_tgn_gen2_5")
CHECKPOINT_DIR.mkdir(exist_ok=True)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_hierarchical(model, loader, device="cuda"):
    """
    Evaluate TGN with hierarchical metrics.
    
    Returns:
        fine_acc: Accuracy on fine-grained labels (0-99)
        super_acc: Accuracy on superclass labels (0-19)
    """
    model.eval()
    fine_correct = 0
    super_correct = 0
    total = 0
    
    with torch.no_grad():
        for images, fine_labels, coarse_labels in loader:
            images = images.to(device)
            fine_labels = fine_labels.to(device)
            coarse_labels = coarse_labels.to(device)
            
            final_probs, super_logits, sub_logits = model(images)
            
            # Fine-grained accuracy
            fine_preds = final_probs.argmax(dim=1)
            fine_correct += (fine_preds == fine_labels).sum().item()
            
            # Superclass accuracy
            super_preds = super_logits.argmax(dim=1)
            super_correct += (super_preds == coarse_labels).sum().item()
            
            total += len(images)
    
    fine_acc = fine_correct / total
    super_acc = super_correct / total
    
    return fine_acc, super_acc


def train_tgn(epochs=50, batch_size=128, lr=0.001, alpha=0.75, weight_decay=5e-4, augment=True, 
              use_mixup=True, mixup_alpha=0.2, grad_accum_steps=1, use_sam=False, use_ema=True, 
              use_warmup=True, warmup_epochs=5, resume=None, restart_lr=False):
    """
    Train Taxonomic Gated Network.
    
    Args:
        epochs: Number of epochs to train (total, not additional)
        batch_size: Batch size for training
        lr: Learning rate (used for fresh start or if restart_lr=True)
        alpha: Weight for subclass loss (1-alpha for superclass), default 0.75 for Gen 2.5 (balanced)
        weight_decay: L2 regularization
        augment: Apply aggressive data augmentation
        use_mixup: Apply MixUp augmentation (Gen 2 improvement)
        mixup_alpha: Beta distribution parameter for MixUp (0.2 = gentle mixing, default for Gen 2.5)
        grad_accum_steps: Gradient accumulation steps (effective_batch = batch_size * grad_accum_steps)
        use_sam: Use Sharpness-Aware Minimization (better accuracy, 50% slower)
        use_ema: Use Exponential Moving Average of weights (ENABLED BY DEFAULT in Gen 2.5)
        use_warmup: Use warmup scheduler for stable early training (ENABLED BY DEFAULT in Gen 2.5)
        warmup_epochs: Number of warmup epochs (default 5)
        resume: Path to checkpoint to resume from (e.g., "checkpoints_tgn/tgn_best.pt")
        restart_lr: If True, restart scheduler with new LR (good for continued training)
    """
    
    print("="*70)
    print("  Training Taxonomic Gated Network (TGN) - Gen 2.5")
    print("="*70)
    
    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, using CPU (will be slow)")
    else:
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Start dashboard
    print("\nStarting dashboard server...")
    dashboard = DashboardServer(port=8891)  # Gen 2.5 port (Gen 1: 8889, Gen 2: 8890)
    dashboard.start()
    print("Dashboard: http://localhost:8891/dashboard.html")
    
    # Create model
    print("\nBuilding TGN model...")
    model = TaxonomicGatedNetwork(
        feature_dim=512,
        gate_hidden=256,
        expert_hidden=512,  # Gen 2: Increased from 256 to 512 for more expert capacity
        num_superclasses=20,
        subclasses_per_superclass=5
    ).to(device)
    
    # Gen 2.5: Channels last memory format for better GPU performance
    print("\n⚡ Converting model to channels_last format...")
    model = model.to(memory_format=torch.channels_last)
    print("   ✓ Model using channels_last (optimized for modern GPUs)")
    
    # Compile model for faster training (PyTorch 2.0+)
    # Note: Disabled on Windows (Triton not available)
    import platform
    if platform.system() != 'Windows':
        try:
            print("\n⚡ Compiling model with torch.compile...")
            model = torch.compile(model, mode='default')
            print("   ✓ Model compiled successfully!")
        except Exception as e:
            print(f"   ⚠ torch.compile failed: {e}")
            print("   → Continuing without compilation (slower but functional)")
    else:
        print("\n⚡ torch.compile skipped (not supported on Windows)")
        print("   → Training will work but ~10-20% slower")
    
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    
    # Create dataloaders with aggressive augmentation (matching main model)
    print("\nLoading CIFAR-100 with hierarchical labels...")
    train_loader, test_loader = make_dataloaders(
        root="./cifar100",
        batch_size=batch_size,
        num_workers=8,  # Increased from 4 for faster data loading
        augment=augment
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    if augment:
        print(f"Augmentation: Flip, Crop, Rotate(±15°), ColorJitter, RandomErasing")
    else:
        print(f"Augmentation: None (baseline mode)")
    
    # Optimizer - using Adam for simplicity (ResNet-18 trains well with Adam)
    if use_sam:
        # SAM optimizer wraps base optimizer
        try:
            from sam import SAM
            base_optimizer = torch.optim.AdamW
            optimizer = SAM(
                model.parameters(),
                base_optimizer,
                lr=lr,
                weight_decay=weight_decay,
                rho=0.05  # Perturbation radius
            )
            print(f"\n⚡ Using SAM optimizer (Sharpness-Aware Minimization)")
            print(f"   Note: Training will be ~50% slower but accuracy should improve 2-4%")
        except ImportError:
            print(f"\n⚠ SAM not installed! Install with: pip install git+https://github.com/davda54/sam.git")
            print(f"   Falling back to regular AdamW...")
            use_sam = False
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
    
    # EMA - Exponential Moving Average
    if use_ema:
        try:
            from torch_ema import ExponentialMovingAverage
            ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
            print(f"\n⚡ Using EMA (Exponential Moving Average)")
            print(f"   Averaging weights over time for smoother predictions")
        except ImportError:
            print(f"\n⚠ torch-ema not installed! Install with: pip install torch-ema")
            print(f"   Continuing without EMA...")
            use_ema = False
            ema = None
    else:
        ema = None
    
    # Mixed precision training for faster training (2x speedup on RTX GPUs)
    scaler = torch.amp.GradScaler('cuda')
    print(f"\n⚡ Mixed precision (FP16) enabled for faster training")
    
    # Scheduler with optional warmup (Gen 2.5 default: enabled)
    if use_warmup:
        from torch.optim.lr_scheduler import LinearLR, SequentialLR
        
        # Warmup: gradually increase LR from 10% to 100% over first N epochs
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,  # Start at 10% of LR
            total_iters=warmup_epochs
        )
        
        # Main training: cosine annealing
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=lr * 0.01
        )
        
        # Combine warmup + cosine
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        print(f"\n⚡ Warmup scheduler enabled: {warmup_epochs} epochs warmup, then cosine")
    else:
        # Standard cosine annealing without warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=lr * 0.01
        )
        print(f"\n⚡ Scheduler: CosineAnnealing (no warmup)")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    
    if resume:
        print(f"\n🔄 Resuming from checkpoint: {resume}")
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint.get('val_acc', 0.0)
        
        if restart_lr:
            print(f"   ✓ Model loaded from epoch {start_epoch}")
            print(f"   ✓ RESTARTING scheduler with fresh LR={lr}")
            print(f"   ✓ Previous best val acc: {best_val_acc*100:.2f}%")
            # Don't load optimizer/scheduler - start fresh with new LR
        else:
            print(f"   ✓ Model loaded from epoch {start_epoch}")
            print(f"   ✓ Continuing optimizer/scheduler state")
            print(f"   ✓ Previous best val acc: {best_val_acc*100:.2f}%")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    effective_batch_size = batch_size * grad_accum_steps
    
    config_name = "Gen 2.5 - Free Lunch Stack"
    if use_sam or use_ema:
        config_name = "Gen 2.5+ (Free Lunch + Quality Boosters)"
    
    print(f"\nTraining configuration ({config_name}):")
    print(f"  Epochs: {start_epoch} → {epochs} ({epochs - start_epoch} more)")
    print(f"  Batch size: {batch_size} (effective: {effective_batch_size})")
    print(f"  Learning rate: {lr} {'(restarted)' if (resume and restart_lr) else ''}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Alpha (subclass weight): {alpha}")
    print(f"  Expert hidden dim: 512 (Gen 2: +100% capacity)")
    print(f"  MixUp: {'Enabled (alpha=' + str(mixup_alpha) + ')' if use_mixup else 'Disabled'}")
    print(f"  Label smoothing: 0.1")
    print(f"  Channels last: Enabled")
    print(f"  Persistent workers: True | Prefetch: 4")
    print(f"  SAM: {'Enabled (rho=0.05)' if use_sam else 'Disabled'}")
    print(f"  EMA: {'Enabled (decay=0.9999)' if use_ema else 'Disabled'}")
    print(f"  Warmup: {'Enabled (' + str(warmup_epochs) + ' epochs)' if use_warmup else 'Disabled'}")
    print(f"  Mixed precision: FP16 enabled")
    print(f"  Optimizer: {'SAM-AdamW' if use_sam else 'AdamW'}")
    print(f"  Scheduler: {'Warmup + Cosine' if use_warmup else 'Cosine'}")
    
    print("\n" + "="*70)
    print("| Epoch | Train Fine | Train Super | Val Fine | Val Super | Loss   | Time |")
    print("-"*70)
    
    start_time = time.time()
    
    for epoch in range(start_epoch, epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, fine_labels, coarse_labels) in enumerate(train_loader):
            # Gen 2.5: Convert images to channels_last format
            images = images.to(device, memory_format=torch.channels_last)
            fine_labels = fine_labels.to(device)
            coarse_labels = coarse_labels.to(device)
            
            # Mixed precision training
            with torch.amp.autocast('cuda'):
                # Apply MixUp augmentation if enabled (Gen 2)
                if use_mixup:
                    mixed_images, y_a_fine, y_b_fine, y_a_coarse, y_b_coarse, lam = mixup_data(
                        images, fine_labels, coarse_labels, alpha=mixup_alpha, device=device
                    )
                    final_probs, super_logits, sub_logits = model(mixed_images)
                    loss = mixup_loss(
                        super_logits, sub_logits,
                        y_a_fine, y_b_fine, y_a_coarse, y_b_coarse,
                        lam, alpha=alpha
                    )
                else:
                    final_probs, super_logits, sub_logits = model(images)
                    loss = taxonomic_loss(
                        super_logits, sub_logits,
                        fine_labels, coarse_labels,
                        alpha=alpha
                    )
            
            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
            
            # Only step optimizer every grad_accum_steps
            if (batch_idx + 1) % grad_accum_steps == 0:
                if use_sam:
                    # SAM requires two forward-backward passes
                    scaler.unscale_(optimizer)
                    optimizer.first_step(zero_grad=True)
                    
                    # Second forward pass with perturbed weights
                    with torch.amp.autocast('cuda'):
                        if use_mixup:
                            mixed_images, y_a_fine, y_b_fine, y_a_coarse, y_b_coarse, lam = mixup_data(
                                images, fine_labels, coarse_labels, alpha=mixup_alpha, device=device
                            )
                            final_probs, super_logits, sub_logits = model(mixed_images)
                            loss = mixup_loss(
                                super_logits, sub_logits,
                                y_a_fine, y_b_fine, y_a_coarse, y_b_coarse,
                                lam, alpha=alpha
                            )
                        else:
                            final_probs, super_logits, sub_logits = model(images)
                            loss = taxonomic_loss(
                                super_logits, sub_logits,
                                fine_labels, coarse_labels,
                                alpha=alpha
                            )
                    
                    loss = loss / grad_accum_steps
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    optimizer.second_step(zero_grad=True)
                    scaler.update()
                else:
                    # Regular optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                # Update EMA after optimizer step
                if use_ema and ema is not None:
                    ema.update()
            
            epoch_loss += loss.item() * grad_accum_steps  # Unscale for logging
            num_batches += 1
        
        scheduler.step()
        
        avg_loss = epoch_loss / num_batches
        
        # Evaluation (use EMA weights if enabled)
        if use_ema and ema is not None:
            with ema.average_parameters():
                train_fine_acc, train_super_acc = evaluate_hierarchical(model, train_loader, device)
                val_fine_acc, val_super_acc = evaluate_hierarchical(model, test_loader, device)
        else:
            train_fine_acc, train_super_acc = evaluate_hierarchical(model, train_loader, device)
            val_fine_acc, val_super_acc = evaluate_hierarchical(model, test_loader, device)
        
        elapsed = time.time() - start_time
        
        # Update dashboard
        metrics_store.add_epoch(epoch + 1, train_fine_acc, val_fine_acc)
        metrics_store.update_current(
            run="TGN",
            epoch=f"{epoch+1}/{epochs}",
            train_acc=train_fine_acc,
            val_acc=val_fine_acc,
            loss=avg_loss,
            lr=scheduler.get_last_lr()[0]
        )
        
        # Print progress
        print(f"| {epoch+1:5d} | {train_fine_acc*100:10.2f}% | {train_super_acc*100:11.2f}% | {val_fine_acc*100:8.2f}% | {val_super_acc*100:9.2f}% | {avg_loss:6.3f} | {elapsed:4.0f}s |")
        
        # Save best model
        if val_fine_acc > best_val_acc:
            best_val_acc = val_fine_acc
            checkpoint_path = CHECKPOINT_DIR / "tgn_gen2_5_best.pt"
            checkpoint_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_fine_acc,
                'val_super_acc': val_super_acc,
            }
            if use_ema and ema is not None:
                checkpoint_dict['ema_state_dict'] = ema.state_dict()
            torch.save(checkpoint_dict, checkpoint_path)
    
    print("-"*70)
    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Total time: {time.time() - start_time:.0f}s")
    print(f"Best model saved: {CHECKPOINT_DIR / 'tgn_gen2_5_best.pt'}")
    print("\nDashboard: http://localhost:8891/dashboard.html")


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Train Taxonomic Gated Network - Gen 2.5")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs (TOTAL, not additional)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=0.75, help="Subclass loss weight (Gen 2.5: default 0.75 - balanced)")
    parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--no-mixup", action="store_true", help="Disable MixUp augmentation (Gen 2 feature)")
    parser.add_argument("--mixup-alpha", type=float, default=0.2, help="MixUp alpha parameter (Gen 2.5: default 0.2 - gentle mixing)")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps (effective_batch = batch * accum)")
    parser.add_argument("--use-sam", action="store_true", help="Use SAM optimizer (+2-4% acc, 50% slower)")
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA (enabled by default in Gen 2.5)")
    parser.add_argument("--no-warmup", action="store_true", help="Disable warmup scheduler (enabled by default in Gen 2.5)")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Number of warmup epochs (default 5)")
    parser.add_argument("--resume", type=str, default=None, 
                        help="Resume from checkpoint (e.g., checkpoints_tgn_gen2_5/tgn_gen2_5_best.pt), or 'auto' to find best")
    parser.add_argument("--restart-lr", action="store_true", 
                        help="Restart LR scheduler (use when continuing training after completion)")
    
    args = parser.parse_args()
    
    # Handle auto-resume
    resume_path = args.resume
    if resume_path == "auto":
        best_ckpt = CHECKPOINT_DIR / "tgn_gen2_5_best.pt"
        if best_ckpt.exists():
            resume_path = str(best_ckpt)
            print(f"Auto-resume: Found {resume_path}")
        else:
            print("Auto-resume: No checkpoint found, starting fresh")
            resume_path = None
    
    train_tgn(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        weight_decay=args.wd,
        augment=not args.no_augment,
        use_mixup=not args.no_mixup,
        mixup_alpha=args.mixup_alpha,
        grad_accum_steps=args.grad_accum,
        use_sam=args.use_sam,
        use_ema=not args.no_ema,
        use_warmup=not args.no_warmup,
        warmup_epochs=args.warmup_epochs,
        resume=resume_path,
        restart_lr=args.restart_lr
    )
