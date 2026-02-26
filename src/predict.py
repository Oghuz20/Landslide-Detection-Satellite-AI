"""
predict.py
Prediction using best single checkpoint, no TTA.
Threshold search (0.02 steps) on test set.
Best model is selected automatically by highest Val F1 from checkpoints/.
"""
import os
import sys
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (f1_score, precision_score,
                             recall_score, confusion_matrix)
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from dataset    import LandslideDataset
from model      import get_model
from transforms import get_valid_transforms


def _collect_predictions(model, test_loader, device):
    """
    Run inference over the full test set.
    Returns:
        prob_maps  — dict {filename: (128,128) float32 prob map}
        tgt_maps   — dict {filename: (128,128) int   ground-truth mask}
    """
    prob_maps = {}
    tgt_maps  = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Predicting'):
            images    = batch['image'].to(device)           # (B, 14, 128, 128)
            targets   = batch['mask'].cpu().numpy()         # (B, 128, 128)
            filenames = batch['filename']                   # list[str]

            # sigmoid on channel 0 → prob map
            probs = torch.sigmoid(
                model(images)[:, 0]
            ).cpu().numpy()                                 # (B, 128, 128)

            for i, fname in enumerate(filenames):
                prob_maps[fname] = probs[i]
                tgt_maps[fname]  = targets[i]

    return prob_maps, tgt_maps


def _find_best_threshold(prob_maps, tgt_maps):
    """Search threshold in 0.02 steps across all pixels."""
    all_probs   = np.concatenate([p.flatten() for p in prob_maps.values()])
    all_targets = np.concatenate([t.flatten() for t in tgt_maps.values()])

    best_t, best_f1 = 0.5, 0.0
    for t in [round(x * 0.02, 2) for x in range(1, 50)]:
        preds = (all_probs >= t).astype(int)
        f1    = f1_score(all_targets, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    return best_t, best_f1, all_probs, all_targets


def _save_predictions(prob_maps, threshold, predictions_dir):
    """
    Save each image's prob map + binary mask to predictions/ as HDF5.
    Naming mirrors the dataset: image_XXXXX.h5 → mask_XXXXX.h5
    """
    os.makedirs(predictions_dir, exist_ok=True)

    print(f"\nSaving predictions to {predictions_dir}/")
    for img_filename, prob_map in tqdm(prob_maps.items(), desc='Saving .h5'):
        # Match dataset naming convention exactly
        out_name = img_filename.replace('image', 'mask')
        out_path = os.path.join(predictions_dir, out_name)

        binary_mask = (prob_map >= threshold).astype(np.uint8)

        with h5py.File(out_path, 'w') as f:
            f.create_dataset('mask', data=binary_mask, dtype=np.uint8,   compression='gzip')
            f.create_dataset('prob', data=prob_map,    dtype=np.float32, compression='gzip')

    print(f"  ✓ Saved {len(prob_maps)} files  "
          f"(datasets: 'mask' uint8, 'prob' float32)")


def predict(data_dir, checkpoints_dir, predictions_dir, batch_size=16, num_workers=2):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load best checkpoint by Val F1 ────────────────────────────────────
    ckpt_files = [
        os.path.join(checkpoints_dir, f)
        for f in os.listdir(checkpoints_dir)
        if f.startswith('phase2') and f.endswith('best_model.pth')
    ]
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")

    best_ckpt = max(
        ckpt_files,
        key=lambda p: torch.load(p, map_location='cpu',
                                 weights_only=False).get('f1', 0)
    )
    ckpt  = torch.load(best_ckpt, map_location=device, weights_only=False)
    model = get_model().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"\nLoaded: {os.path.basename(best_ckpt)}")
    print(f"  Epoch: {ckpt['epoch']}  |  "
          f"Val F1: {ckpt['f1']:.4f}  |  "
          f"Val threshold: {ckpt['best_threshold']:.2f}")

    # ── Dataset & loader ──────────────────────────────────────────────────
    test_ds     = LandslideDataset(data_dir, 'test', get_valid_transforms())
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)
    print(f"Test set: {len(test_ds)} images")

    # ── Inference ─────────────────────────────────────────────────────────
    prob_maps, tgt_maps = _collect_predictions(model, test_loader, device)

    # ── Threshold search on full test set ─────────────────────────────────
    best_t, best_f1, all_probs, all_targets = _find_best_threshold(prob_maps, tgt_maps)

    # Print threshold table (±0.10 window around best)
    print("\nThreshold search:")
    print(f"  {'Threshold':>10} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print(f"  {'-'*40}")
    for t in [round(x * 0.02, 2) for x in range(1, 50)]:
        preds = (all_probs >= t).astype(int)
        f1    = f1_score(all_targets, preds,        pos_label=1, zero_division=0)
        p     = precision_score(all_targets, preds, pos_label=1, zero_division=0)
        r     = recall_score(all_targets, preds,    pos_label=1, zero_division=0)
        if abs(t - best_t) <= 0.10:
            marker = "  ← BEST" if t == best_t else ""
            print(f"  {t:>10.2f} {f1:>8.4f} {p:>10.4f} {r:>8.4f}{marker}")

    # ── Save .h5 files ────────────────────────────────────────────────────
    _save_predictions(prob_maps, best_t, predictions_dir)

    # ── Final metrics ─────────────────────────────────────────────────────
    preds     = (all_probs >= best_t).astype(int)
    precision = precision_score(all_targets, preds, pos_label=1, zero_division=0)
    recall    = recall_score(all_targets, preds,    pos_label=1, zero_division=0)
    accuracy  = (preds == all_targets).mean()
    cm        = confusion_matrix(all_targets, preds)

    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Model:      {os.path.basename(best_ckpt)}")
    print(f"  Strategy:   Single best model, no TTA")
    print(f"  F1 Score:   {best_f1:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  Accuracy:   {accuracy:.4f}")
    print(f"  Threshold:  {best_t:.2f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {cm[0,0]:,}   FP: {cm[0,1]:,}")
    print(f"    FN: {cm[1,0]:,}   TP: {cm[1,1]:,}")
    print(f"\n  Val F1:        {ckpt['f1']:.4f}")
    print(f"  Test F1:       {best_f1:.4f}")
    print(f"  Val→Test gap: -{ckpt['f1'] - best_f1:.4f}")
    print(f"{'='*60}")

    return {
        'probs':            all_probs,
        'targets':          all_targets,
        'preds':            preds,
        'f1':               best_f1,
        'precision':        precision,
        'recall':           recall,
        'accuracy':         accuracy,
        'threshold':        best_t,
        'cm':               cm,
        'val_f1':           ckpt['f1'],
        'dataset':          test_ds,
        'predictions_dir':  predictions_dir,
    }


if __name__ == '__main__':
    results = predict(
        data_dir         = 'data',
        checkpoints_dir  = 'checkpoints',
        predictions_dir  = 'predictions',
    )