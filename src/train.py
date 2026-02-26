"""
Two-phase training:
  Phase 1 — Train on TrainData only, validate on ValidData, early stopping
  Phase 2 — Retrain on Train+Valid combined, fixed 35 epochs, no early stopping
Top-3 checkpoints saved by Val F1 in both phases.
Hyperparameters loaded from configs/optuna_best_params.json if present.
"""

import os
import json
import heapq
import torch
import torch.optim as optim
import numpy as np
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from dataset    import LandslideDataset
from model      import get_model
from losses     import WeightedBCEDiceLoss
from transforms import get_train_transforms, get_valid_transforms


# ── Optuna params loader ───────────────────────────────────────────────────

def load_optuna_params(config_path):
    """
    Loads lr, wd, dice_w from Optuna JSON.
    Applies LR guard: random init (encoder_weights=None) is unstable outside [1e-4, 3e-4].
    Falls back to safe defaults if file is missing or keys are absent.
    """
    defaults = {'lr': 0.0002, 'wd': 0.00005, 'dice_w': 0.7}

    if os.path.exists(config_path):
        with open(config_path) as f:
            p = json.load(f)
        lr     = p['lr']
        wd     = p['wd']
        dice_w = p['dice_w']
        print(f"[Config] Loaded Optuna params from {config_path}")
        print(f"         lr={lr:.6f}  wd={wd:.6f}  dice_w={dice_w:.3f}")
    else:
        lr, wd, dice_w = defaults['lr'], defaults['wd'], defaults['dice_w']
        print(f"[Config] {config_path} not found — using defaults "
              f"lr={lr}  wd={wd}  dice_w={dice_w}")

    # Guard: random init needs stable LR between 1e-4 and 3e-4
    if lr < 0.0001 or lr > 0.0003:
        print(f"[Config] LR {lr:.6f} outside safe range [1e-4, 3e-4] → reset to 0.0002")
        lr = 0.0002

    print(f"[Config] Final params: lr={lr:.6f}  wd={wd:.6f}  dice_w={dice_w:.3f}")
    return lr, wd, dice_w


# ── Validation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_probs, all_targets = [], []
    for batch in loader:
        images  = batch['image'].to(device)
        masks   = batch['mask'].to(device).float()
        outputs = model(images)
        probs   = torch.sigmoid(outputs[:, 0]).cpu().numpy().flatten()
        targets = masks.cpu().numpy().flatten()
        all_probs.extend(probs)
        all_targets.extend(targets)

    all_probs   = np.array(all_probs)
    all_targets = np.array(all_targets)

    best_t, best_f1 = 0.5, 0.0
    for t in [round(x * 0.02, 2) for x in range(1, 50)]:
        preds = (all_probs >= t).astype(int)
        f1    = f1_score(all_targets, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    preds     = (all_probs >= best_t).astype(int)
    precision = precision_score(all_targets, preds, pos_label=1, zero_division=0)
    recall    = recall_score(all_targets, preds,    pos_label=1, zero_division=0)
    cm        = confusion_matrix(all_targets, preds)

    return {
        'f1': best_f1, 'precision': precision, 'recall': recall,
        'accuracy': (preds == all_targets).mean(),
        'confusion_matrix': cm, 'best_threshold': best_t
    }


def _run_loop(model, train_loader, valid_loader, optimizer, scheduler,
              criterion, scaler, device, use_amp, num_epochs,
              save_dir, patience, phase_name, top_k=3):
    """Shared training loop used by both phases."""

    os.makedirs(save_dir, exist_ok=True)
    best_f1           = 0.0
    epochs_no_improve = 0
    top3              = []
    history           = {'train_loss': [], 'valid_f1': [],
                         'valid_precision': [], 'valid_recall': [], 'lr': []}
    start_time        = datetime.now()

    for epoch in range(1, num_epochs + 1):
        print(f"\n[{phase_name}] Epoch {epoch}/{num_epochs}")
        print("-" * 70)

        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc='Training')

        for batch in pbar:
            images = batch['image'].to(device)
            masks  = batch['mask'].to(device).float()
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss    = criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss    = criterion(outputs, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss = total_loss / len(train_loader)
        scheduler.step()

        torch.cuda.empty_cache()
        metrics    = validate(model, valid_loader, device)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['valid_f1'].append(metrics['f1'])
        history['valid_precision'].append(metrics['precision'])
        history['valid_recall'].append(metrics['recall'])
        history['lr'].append(current_lr)

        cm = metrics['confusion_matrix']
        print(f"  Loss: {train_loss:.4f}  |  LR: {current_lr:.8f}")
        print(f"  Val F1: {metrics['f1']:.4f}  "
              f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
              f"thr={metrics['best_threshold']:.2f}")
        print(f"  TN:{cm[0,0]:,}  FP:{cm[0,1]:,}  "
              f"FN:{cm[1,0]:,}  TP:{cm[1,1]:,}")

        if metrics['f1'] > best_f1:
            best_f1           = metrics['f1']
            epochs_no_improve = 0
            print(f"  ✓ NEW BEST F1={best_f1:.4f}")
        else:
            epochs_no_improve += 1
            if patience < num_epochs:
                print(f"  No improvement {epochs_no_improve}/{patience}")
            else:
                print(f"  (best so far: {best_f1:.4f}) — no early stopping")

        # Save top-3 by Val F1
        ckpt_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch:02d}.pth')
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'f1':                   metrics['f1'],
            'best_threshold':       metrics['best_threshold'],
            'train_loss':           train_loss,
            'metrics':              metrics,
        }, ckpt_path)

        heapq.heappush(top3, (metrics['f1'], epoch, ckpt_path))
        if len(top3) > top_k:
            worst_f1, worst_ep, worst_path = heapq.heappop(top3)
            if os.path.exists(worst_path):
                os.remove(worst_path)
            print(f"  (dropped ep{worst_ep:02d} F1={worst_f1:.4f})")

        top3_str = " | ".join(
            f"ep{e:02d}(F1={f:.4f})"
            for f, e, _ in sorted(top3, key=lambda x: -x[0])
        )
        print(f"  Top-3: {top3_str}")

        elapsed   = (datetime.now() - start_time).total_seconds() / 3600
        remaining = (elapsed / epoch) * (num_epochs - epoch)
        print(f"  Time: {elapsed:.1f}h elapsed  ~{remaining:.1f}h remaining")

        if patience < num_epochs and epochs_no_improve >= patience:
            print(f"\n⚠ Early stopping at epoch {epoch}")
            break

    total_time = (datetime.now() - start_time).total_seconds() / 3600
    print(f"\n{'='*70}")
    print(f"[{phase_name}] DONE  |  {total_time:.2f}h  |  Best Val F1: {best_f1:.4f}")
    for f1v, ep, path in sorted(top3, key=lambda x: -x[0]):
        print(f"  epoch_{ep:02d}  F1={f1v:.4f}  →  {os.path.basename(path)}")
    print(f"{'='*70}")

    return history, top3


def train_phase1(data_dir, save_dir, lr=0.0002, weight_decay=0.00005,
                 pos_weight=12.0, dice_w=0.7, num_epochs=60,
                 batch_size=16, num_workers=2, patience=15):
    """Phase 1: Train on TrainData only, validate on ValidData, with early stopping."""

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()

    train_ds = LandslideDataset(data_dir, 'train', get_train_transforms())
    valid_ds = LandslideDataset(data_dir, 'valid', get_valid_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    model     = get_model().to(device)
    criterion = WeightedBCEDiceLoss(pos_weight=pos_weight, dice_w=dice_w)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    return _run_loop(
        model, train_loader, valid_loader, optimizer, scheduler,
        criterion, scaler, device, use_amp,
        num_epochs, save_dir, patience, "Phase1-TrainOnly"
    )


def train_phase2(data_dir, save_dir, lr=0.0002, weight_decay=0.00005,
                 pos_weight=12.0, dice_w=0.7, num_epochs=35,
                 batch_size=16, num_workers=2):
    """Phase 2: Retrain on Train+Valid combined, fixed epochs, no early stopping."""

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()

    train_ds     = LandslideDataset(data_dir, 'train', get_train_transforms())
    valid_ds_aug = LandslideDataset(data_dir, 'valid', get_train_transforms())
    valid_ds     = LandslideDataset(data_dir, 'valid', get_valid_transforms())
    combined_ds  = ConcatDataset([train_ds, valid_ds_aug])

    train_loader = DataLoader(combined_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"Combined dataset: {len(combined_ds)} images  |  "
          f"Batches/epoch: {len(train_loader)}")

    model     = get_model().to(device)
    criterion = WeightedBCEDiceLoss(pos_weight=pos_weight, dice_w=dice_w)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=num_epochs, T_mult=1, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # patience=num_epochs effectively disables early stopping
    return _run_loop(
        model, train_loader, valid_loader, optimizer, scheduler,
        criterion, scaler, device, use_amp,
        num_epochs, save_dir, num_epochs, "Phase2-Combined"
    )


if __name__ == '__main__':
    DATA_DIR    = 'data'
    SAVE_DIR    = 'checkpoints'
    CONFIG_PATH = 'configs/optuna_best_params.json'

    lr, wd, dice_w = load_optuna_params(CONFIG_PATH)

    print("=" * 70)
    print("PHASE 1 — Train on TrainData, validate on ValidData")
    print("=" * 70)
    history1, top3_phase1 = train_phase1(
        DATA_DIR, SAVE_DIR,
        lr=lr, weight_decay=wd, dice_w=dice_w
    )

    print("\n" + "=" * 70)
    print("PHASE 2 — Retrain on Train+Valid combined")
    print("=" * 70)
    history2, top3_phase2 = train_phase2(
        DATA_DIR, SAVE_DIR,
        lr=lr, weight_decay=wd, dice_w=dice_w
    )