"""
Local Evaluation Script
For testing the Attention U-Net model trained on Colab
"""

import os
import sys
import torch
import h5py
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, 'src')
from dataset import LandslideDataset, get_valid_transforms
from torch.utils.data import DataLoader

def create_attention_unet():
    """Recreate the Attention U-Net architecture"""
    import torch.nn as nn
    
    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels, mid_channels=None):
            super().__init__()
            if not mid_channels:
                mid_channels = out_channels

            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),  # 0
                nn.BatchNorm2d(mid_channels),                                                   # 1
                nn.ReLU(inplace=True),                                                          # 2
                nn.Dropout2d(0.3),                                                              # 3  ← IMPORTANT
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),   # 4
                nn.BatchNorm2d(out_channels),                                                   # 5
                nn.ReLU(inplace=True)                                                           # 6
            )

        def forward(self, x):
            return self.double_conv(x)


    
    class Down(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )
        def forward(self, x):
            return self.maxpool_conv(x)
    
    class AttentionBlock(nn.Module):
        def __init__(self, F_g, F_l, F_int):
            super().__init__()
            self.W_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
                nn.BatchNorm2d(F_int)
            )
            self.W_x = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
                nn.BatchNorm2d(F_int)
            )
            self.psi = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )
            self.relu = nn.ReLU(inplace=True)
        
        def forward(self, g, x):
            g1 = self.W_g(g)
            x1 = self.W_x(x)
            psi = self.relu(g1 + x1)
            psi = self.psi(psi)
            return x * psi
    
    class AttentionUNet(nn.Module):
        def __init__(self, n_channels=14, n_classes=2):
            super().__init__()
            
            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            self.down4 = Down(512, 1024)
            
            self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
            self.att1 = AttentionBlock(F_g=512, F_l=512, F_int=256)
            self.upconv1 = DoubleConv(1024, 512)
            
            self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.att2 = AttentionBlock(F_g=256, F_l=256, F_int=128)
            self.upconv2 = DoubleConv(512, 256)
            
            self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
            self.upconv3 = DoubleConv(256, 128)
            
            self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.att4 = AttentionBlock(F_g=64, F_l=64, F_int=32)
            self.upconv4 = DoubleConv(128, 64)
            
            self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        
        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            
            d1 = self.up1(x5)
            x4 = self.att1(g=d1, x=x4)
            d1 = torch.cat((x4, d1), dim=1)
            d1 = self.upconv1(d1)
            
            d2 = self.up2(d1)
            x3 = self.att2(g=d2, x=x3)
            d2 = torch.cat((x3, d2), dim=1)
            d2 = self.upconv2(d2)
            
            d3 = self.up3(d2)
            x2 = self.att3(g=d3, x=x2)
            d3 = torch.cat((x2, d3), dim=1)
            d3 = self.upconv3(d3)
            
            d4 = self.up4(d3)
            x1 = self.att4(g=d4, x=x1)
            d4 = torch.cat((x1, d4), dim=1)
            d4 = self.upconv4(d4)
            
            return self.outc(d4)
    
    return AttentionUNet(n_channels=14, n_classes=2)


@torch.no_grad()


def evaluate_predictions(pred_dir, ground_truth_dir):
    """Evaluate predictions"""
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.h5')])
    
    all_preds = []
    all_targets = []
    
    print(f"Loading {len(pred_files)} predictions...")
    for pred_file in tqdm(pred_files):
        # Load prediction
        with h5py.File(os.path.join(pred_dir, pred_file), 'r') as f:
            pred = f['mask'][:]
        
        # Load ground truth
        with h5py.File(os.path.join(ground_truth_dir, pred_file), 'r') as f:
            target = f['mask'][:]
        
        all_preds.extend(pred.flatten())
        all_targets.extend(target.flatten())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    precision = precision_score(all_targets, all_preds, pos_label=1, zero_division=0)
    recall = recall_score(all_targets, all_preds, pos_label=1, zero_division=0)
    f1 = f1_score(all_targets, all_preds, pos_label=1, zero_division=0)
    accuracy = (all_preds == all_targets).mean()
    cm = confusion_matrix(all_targets, all_preds)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'confusion_matrix': cm
    }

def generate_probabilities_attention(checkpoint_path, dataloader, device):
    """Generate probability maps from Attention U-Net checkpoint"""

    model = create_attention_unet()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    predictions = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=os.path.basename(checkpoint_path)):
            images = batch['image'].to(device)
            filenames = batch['filename']

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

            for i, filename in enumerate(filenames):
                predictions[filename] = probs[i]

    return predictions
def ensemble_attention(checkpoint_paths, dataloader, device, threshold=0.70):
    """Average predictions from multiple Attention U-Net checkpoints"""

    all_probs = []

    for ckpt in checkpoint_paths:
        print(f"\nProcessing {ckpt}")
        probs = generate_probabilities_attention(ckpt, dataloader, device)
        all_probs.append(probs)

    ensembled = {}

    print("\nAveraging predictions...")
    for filename in tqdm(sorted(all_probs[0].keys()), desc="Ensembling"):
        maps = [p[filename] for p in all_probs]
        avg = np.mean(maps, axis=0)
        ensembled[filename] = (avg > threshold).astype(np.uint8)

    return ensembled
def save_predictions(predictions, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename, mask in tqdm(predictions.items(), desc='Saving'):
        mask_filename = filename.replace('image', 'mask')
        output_path = os.path.join(output_dir, mask_filename)

        with h5py.File(output_path, 'w') as f:
            f.create_dataset('mask', data=mask, dtype='uint8')

    print(f'✓ Saved {len(predictions)} predictions to {output_dir}')


def main():
    print("="*80)
    print("ATTENTION U-NET - LOCAL EVALUATION")
    print("="*80)
    
    # Configuration
    CHECKPOINT_PATH = './checkpoints_attention/best_model.pth'
    DATA_DIR = './data'
    BATCH_SIZE = 4
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load model
    print("\nLoading Attention U-Net...")
    model = create_attention_unet()
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n⚠️ ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Please download best_model.pth from Colab and place it in checkpoints_attention/")
        return
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Validation F1: {checkpoint['f1']:.4f}")
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = LandslideDataset(DATA_DIR, split='test', transform=get_valid_transforms())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Attention ensemble checkpoints
    checkpoints = [
        './checkpoints_attention/best_model.pth',
        './checkpoints_attention/checkpoint_epoch_30.pth',
        './checkpoints_attention/checkpoint_epoch_35.pth',
    ]

    available = [c for c in checkpoints if os.path.exists(c)]

    if len(available) < 2:
        print("⚠ Need at least 2 checkpoints for ensemble!")
        return

    ensembled = ensemble_attention(available, test_loader, device, threshold=0.7)
    save_predictions(ensembled, './predictions/test_attention_ensemble')

        
    # Evaluate
    print("\nEvaluating predictions...")
    metrics = evaluate_predictions(
        './predictions/test_attention_ensemble',
        os.path.join(DATA_DIR, 'TestData', 'mask')
    )
    
    # Print results
    print("\n" + "="*80)
    print("FINAL RESULTS - ATTENTION U-NET")
    print("="*80)
    print(f"\nTest Set (800 images):")
    print(f"  F1 Score:   {metrics['f1']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    
    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:>10,}  |  FP: {cm[0,1]:>10,}")
    print(f"  FN: {cm[1,0]:>10,}  |  TP: {cm[1,1]:>10,}")
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON WITH PREVIOUS MODEL")
    print("="*80)
    print(f"{'Model':<30} {'F1':<10} {'Precision':<10} {'Recall':<10}")
    print("-"*80)
    print(f"{'Previous U-Net':<30} {0.5901:<10.4f} {0.4959:<10.4f} {0.7286:<10.4f}")
    print(f"{'NEW Attention U-Net':<30} {metrics['f1']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f}")
    
    # Calculate improvement
    improvement = ((metrics['f1'] / 0.5901) - 1) * 100
    print("="*80)
    print(f"\nImprovement: {improvement:+.1f}%")
    
    # Gap analysis
    val_f1 = checkpoint['f1']
    gap = abs(val_f1 - metrics['f1'])
    gap_pct = (gap / val_f1) * 100
    
    print(f"\nGeneralization Analysis:")
    print(f"  Validation F1: {val_f1:.4f}")
    print(f"  Test F1:       {metrics['f1']:.4f}")
    print(f"  Gap:           {gap:.4f} ({gap_pct:.1f}%)")
    
    if gap_pct < 5:
        print(f"  Status:        ✓ Excellent generalization!")
    elif gap_pct < 10:
        print(f"  Status:        ✓ Good generalization")
    else:
        print(f"  Status:        ⚠ Some overfitting")
    
    print("\n" + "="*80)
    
    # Save results
    with open('results_attention_unet.txt', 'w') as f:
        f.write("ATTENTION U-NET - FINAL RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Test F1:        {metrics['f1']:.4f}\n")
        f.write(f"Test Precision: {metrics['precision']:.4f}\n")
        f.write(f"Test Recall:    {metrics['recall']:.4f}\n")
        f.write(f"Validation F1:  {val_f1:.4f}\n")
        f.write(f"Gap:            {gap:.4f} ({gap_pct:.1f}%)\n")
        f.write(f"Improvement:    {improvement:+.1f}%\n")
    
    print("✓ Results saved to results_attention_unet.txt")
    print("="*80)


if __name__ == '__main__':
    main()
