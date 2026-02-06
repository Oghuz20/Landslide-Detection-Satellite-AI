"""
Visualization script for landslide detection results
Creates publication-quality plots and overlays
"""
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.dataset import LandslideDataset, get_valid_transforms


class LandslideVisualizer:
    """
    Comprehensive visualization toolkit for landslide detection
    """
    
    def __init__(self, data_dir='./data', pred_dir='./predictions/test_final'):
        """
        Args:
            data_dir: Root directory containing data
            pred_dir: Directory containing predictions
        """
        self.data_dir = data_dir
        self.pred_dir = pred_dir
        self.output_dir = './visualizations'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Color schemes
        self.mask_colors = ListedColormap(['black', 'red'])  # Background, Landslide
        self.overlay_colors = {
            'ground_truth': (0, 1, 0, 0.5),    # Green, 50% transparent
            'prediction': (1, 0, 0, 0.5),       # Red, 50% transparent
            'true_positive': (0, 1, 0, 0.7),    # Green
            'false_positive': (1, 0, 0, 0.7),   # Red
            'false_negative': (1, 1, 0, 0.7)    # Yellow
        }
    
    def load_image_data(self, filename):
        """
        Load image, ground truth, and prediction
        
        Args:
            filename: Image filename (e.g., 'image_1.h5')
            
        Returns:
            dict with 'image', 'ground_truth', 'prediction'
        """
        # Load RGB bands from satellite image (bands 4,3,2 for true color)
        img_path = os.path.join(self.data_dir, 'TestData', 'img', filename)
        with h5py.File(img_path, 'r') as f:
            image = f['img'][:]  # [128, 128, 14]
        
        # Extract RGB (Sentinel-2 bands 4,3,2 = Red, Green, Blue)
        rgb = image[:, :, [3, 2, 1]]  # BGR to RGB
        
        # Normalize RGB for display
        rgb_normalized = np.zeros_like(rgb)
        for i in range(3):
            band = rgb[:, :, i]
            p2, p98 = np.percentile(band, (2, 98))
            band = np.clip(band, p2, p98)
            if p98 - p2 > 0:
                rgb_normalized[:, :, i] = (band - p2) / (p98 - p2)
        
        # Load ground truth
        mask_filename = filename.replace('image', 'mask')
        mask_path = os.path.join(self.data_dir, 'TestData', 'mask', mask_filename)
        with h5py.File(mask_path, 'r') as f:
            ground_truth = f['mask'][:]
        
        # Load prediction
        pred_path = os.path.join(self.pred_dir, mask_filename)
        if os.path.exists(pred_path):
            with h5py.File(pred_path, 'r') as f:
                prediction = f['mask'][:]
        else:
            prediction = np.zeros_like(ground_truth)
            print(f"Warning: No prediction found for {filename}")
        
        return {
            'image': rgb_normalized,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'filename': filename
        }
    
    def plot_single_prediction(self, filename, save=True):
        """
        Create comprehensive visualization for a single image
        
        Shows:
        - Original RGB image
        - Ground truth mask
        - Prediction mask
        - Overlay comparison
        - Error analysis (TP, FP, FN)
        
        Args:
            filename: Image filename
            save: Whether to save the figure
        """
        data = self.load_image_data(filename)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Row 1: RGB, Ground Truth, Prediction
        # RGB Image
        axes[0, 0].imshow(data['image'])
        axes[0, 0].set_title('Satellite Image (RGB)', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Ground Truth
        axes[0, 1].imshow(data['ground_truth'], cmap=self.mask_colors, vmin=0, vmax=1)
        axes[0, 1].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Prediction
        axes[0, 2].imshow(data['prediction'], cmap=self.mask_colors, vmin=0, vmax=1)
        axes[0, 2].set_title('Model Prediction', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2: Overlays and Error Analysis
        # Overlay on RGB
        axes[1, 0].imshow(data['image'])
        # Ground truth in green
        gt_mask = np.ma.masked_where(data['ground_truth'] == 0, data['ground_truth'])
        axes[1, 0].imshow(gt_mask, cmap=ListedColormap(['none', 'lime']), alpha=0.6)
        # Prediction in red
        pred_mask = np.ma.masked_where(data['prediction'] == 0, data['prediction'])
        axes[1, 0].imshow(pred_mask, cmap=ListedColormap(['none', 'red']), alpha=0.4)
        axes[1, 0].set_title('Overlay: Green=GT, Red=Pred', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Error Analysis
        tp = (data['ground_truth'] == 1) & (data['prediction'] == 1)  # True Positive
        fp = (data['ground_truth'] == 0) & (data['prediction'] == 1)  # False Positive
        fn = (data['ground_truth'] == 1) & (data['prediction'] == 0)  # False Negative
        
        error_map = np.zeros((*data['ground_truth'].shape, 3))
        error_map[tp] = [0, 1, 0]      # Green - Correct detection
        error_map[fp] = [1, 0, 0]      # Red - False alarm
        error_map[fn] = [1, 1, 0]      # Yellow - Missed landslide
        
        axes[1, 1].imshow(data['image'])
        axes[1, 1].imshow(error_map, alpha=0.7)
        axes[1, 1].set_title('Error Analysis', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Legend for error map
        tp_patch = mpatches.Patch(color='green', label=f'True Positive ({tp.sum():,} px)')
        fp_patch = mpatches.Patch(color='red', label=f'False Positive ({fp.sum():,} px)')
        fn_patch = mpatches.Patch(color='yellow', label=f'False Negative ({fn.sum():,} px)')
        axes[1, 1].legend(handles=[tp_patch, fp_patch, fn_patch], loc='upper right', fontsize=10)
        
        # Statistics
        total_pixels = data['ground_truth'].size
        gt_positive = data['ground_truth'].sum()
        pred_positive = data['prediction'].sum()
        
        precision = tp.sum() / pred_positive if pred_positive > 0 else 0
        recall = tp.sum() / gt_positive if gt_positive > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        stats_text = f"""
        Image Statistics
        ─────────────────────────
        Total Pixels: {total_pixels:,}
        
        Ground Truth:
          Landslide: {gt_positive:,} ({gt_positive/total_pixels*100:.2f}%)
          Background: {total_pixels - gt_positive:,}
        
        Prediction:
          Landslide: {pred_positive:,} ({pred_positive/total_pixels*100:.2f}%)
          Background: {total_pixels - pred_positive:,}
        
        Performance:
          Precision: {precision:.4f}
          Recall:    {recall:.4f}
          F1 Score:  {f1:.4f}
        
        Pixel-wise Accuracy:
          True Pos:  {tp.sum():,}
          False Pos: {fp.sum():,}
          False Neg: {fn.sum():,}
        """
        
        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                       family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        axes[1, 2].axis('off')
        
        plt.suptitle(f'Landslide Detection Results: {filename}', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save:
            output_path = os.path.join(self.output_dir, f'prediction_{filename.replace(".h5", ".png")}')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {output_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_best_worst_cases(self, n_samples=5):
        """
        Plot best and worst predictions based on F1 score
        
        Args:
            n_samples: Number of best/worst cases to show
        """
        print("Analyzing all predictions to find best/worst cases...")
        
        # Get all prediction files
        pred_files = sorted([f for f in os.listdir(self.pred_dir) if f.endswith('.h5')])
        
        scores = []
        for pred_file in tqdm(pred_files, desc='Computing F1 scores'):
            filename = pred_file.replace('mask', 'image')
            data = self.load_image_data(filename)
            
            tp = ((data['ground_truth'] == 1) & (data['prediction'] == 1)).sum()
            fp = ((data['ground_truth'] == 0) & (data['prediction'] == 1)).sum()
            fn = ((data['ground_truth'] == 1) & (data['prediction'] == 0)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            scores.append((filename, f1))
        
        # Sort by F1 score
        scores.sort(key=lambda x: x[1])
        
        worst_cases = scores[:n_samples]
        best_cases = scores[-n_samples:][::-1]
        
        print(f"\nBest {n_samples} predictions:")
        for filename, f1 in best_cases:
            print(f"  {filename}: F1 = {f1:.4f}")
        
        print(f"\nWorst {n_samples} predictions:")
        for filename, f1 in worst_cases:
            print(f"  {filename}: F1 = {f1:.4f}")
        
        # Plot best cases
        print("\nGenerating visualizations for best cases...")
        for filename, f1 in best_cases:
            self.plot_single_prediction(filename, save=True)
        
        # Plot worst cases
        print("\nGenerating visualizations for worst cases...")
        for filename, f1 in worst_cases:
            self.plot_single_prediction(filename, save=True)
    
    def plot_confusion_matrix_grid(self, n_samples=16):
        """
        Create grid showing examples from confusion matrix categories
        
        Args:
            n_samples: Total number of samples to show (must be multiple of 4)
        """
        print("Creating confusion matrix grid...")
        
        pred_files = sorted([f for f in os.listdir(self.pred_dir) if f.endswith('.h5')])
        
        # Categorize images
        categories = {'TP': [], 'TN': [], 'FP': [], 'FN': []}
        
        for pred_file in tqdm(pred_files, desc='Categorizing'):
            filename = pred_file.replace('mask', 'image')
            data = self.load_image_data(filename)
            
            tp = ((data['ground_truth'] == 1) & (data['prediction'] == 1)).sum()
            tn = ((data['ground_truth'] == 0) & (data['prediction'] == 0)).sum()
            fp = ((data['ground_truth'] == 0) & (data['prediction'] == 1)).sum()
            fn = ((data['ground_truth'] == 1) & (data['prediction'] == 0)).sum()
            
            # Classify based on dominant category
            max_category = max([('TP', tp), ('TN', tn), ('FP', fp), ('FN', fn)], 
                              key=lambda x: x[1])
            categories[max_category[0]].append(filename)
        
        # Sample from each category
        samples_per_category = n_samples // 4
        selected = {}
        for cat in ['TP', 'TN', 'FP', 'FN']:
            selected[cat] = np.random.choice(categories[cat], 
                                            min(samples_per_category, len(categories[cat])), 
                                            replace=False)
        
        # Create grid
        fig, axes = plt.subplots(4, samples_per_category, figsize=(4*samples_per_category, 16))
        
        category_names = {
            'TP': 'True Positive (Correct Detection)',
            'FP': 'False Positive (False Alarm)',
            'FN': 'False Negative (Missed Landslide)',
            'TN': 'True Negative (Correct Background)'
        }
        
        for row, cat in enumerate(['TP', 'FP', 'FN', 'TN']):
            for col, filename in enumerate(selected[cat]):
                data = self.load_image_data(filename)
                
                # Show RGB with overlay
                axes[row, col].imshow(data['image'])
                
                if cat in ['TP', 'FP']:
                    # Show predictions (red)
                    pred_mask = np.ma.masked_where(data['prediction'] == 0, data['prediction'])
                    axes[row, col].imshow(pred_mask, cmap=ListedColormap(['none', 'red']), alpha=0.6)
                
                if cat in ['TP', 'FN']:
                    # Show ground truth (green)
                    gt_mask = np.ma.masked_where(data['ground_truth'] == 0, data['ground_truth'])
                    axes[row, col].imshow(gt_mask, cmap=ListedColormap(['none', 'lime']), alpha=0.4)
                
                axes[row, col].set_title(filename.replace('.h5', ''), fontsize=8)
                axes[row, col].axis('off')
                
                # Add category label on first column
                if col == 0:
                    axes[row, col].text(-0.1, 0.5, category_names[cat], 
                                       transform=axes[row, col].transAxes,
                                       fontsize=12, fontweight='bold',
                                       verticalalignment='center',
                                       rotation=90)
        
        plt.suptitle('Confusion Matrix Examples: Green=Ground Truth, Red=Prediction', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'confusion_matrix_grid.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_performance_distribution(self):
        """
        Plot distribution of F1 scores across all test images
        """
        print("Computing performance distribution...")
        
        pred_files = sorted([f for f in os.listdir(self.pred_dir) if f.endswith('.h5')])
        
        f1_scores = []
        precisions = []
        recalls = []
        
        for pred_file in tqdm(pred_files, desc='Computing metrics'):
            filename = pred_file.replace('mask', 'image')
            data = self.load_image_data(filename)
            
            tp = ((data['ground_truth'] == 1) & (data['prediction'] == 1)).sum()
            fp = ((data['ground_truth'] == 0) & (data['prediction'] == 1)).sum()
            fn = ((data['ground_truth'] == 1) & (data['prediction'] == 0)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # F1 Score Distribution
        axes[0, 0].hist(f1_scores, bins=30, color='green', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(np.mean(f1_scores), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(f1_scores):.4f}')
        axes[0, 0].axvline(np.median(f1_scores), color='blue', linestyle='--', 
                          linewidth=2, label=f'Median: {np.median(f1_scores):.4f}')
        axes[0, 0].set_xlabel('F1 Score', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('F1 Score Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision Distribution
        axes[0, 1].hist(precisions, bins=30, color='blue', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(np.mean(precisions), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(precisions):.4f}')
        axes[0, 1].set_xlabel('Precision', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title('Precision Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Recall Distribution
        axes[1, 0].hist(recalls, bins=30, color='orange', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(np.mean(recalls), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(recalls):.4f}')
        axes[1, 0].set_xlabel('Recall', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('Recall Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Precision-Recall Scatter
        axes[1, 1].scatter(recalls, precisions, alpha=0.5, c=f1_scores, cmap='viridis', s=30)
        axes[1, 1].set_xlabel('Recall', fontsize=12)
        axes[1, 1].set_ylabel('Precision', fontsize=12)
        axes[1, 1].set_title('Precision-Recall Scatter (colored by F1)', 
                            fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
        cbar.set_label('F1 Score', fontsize=10)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'performance_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
        
        # Print statistics
        print(f"\n{'='*60}")
        print("PERFORMANCE STATISTICS")
        print(f"{'='*60}")
        print(f"F1 Score:   Mean={np.mean(f1_scores):.4f}, Std={np.std(f1_scores):.4f}")
        print(f"Precision:  Mean={np.mean(precisions):.4f}, Std={np.std(precisions):.4f}")
        print(f"Recall:     Mean={np.mean(recalls):.4f}, Std={np.std(recalls):.4f}")
        print(f"{'='*60}")


def main():
    """
    Main visualization function
    """
    print("="*80)
    print("LANDSLIDE DETECTION - VISUALIZATION SUITE")
    print("="*80)
    
    # Initialize visualizer
    viz = LandslideVisualizer(
        data_dir='./data',
        pred_dir='./predictions/test_final'
    )
    
    print("\n1. Creating performance distribution plots...")
    viz.plot_performance_distribution()
    
    print("\n2. Finding and plotting best/worst cases...")
    viz.plot_best_worst_cases(n_samples=5)
    
    print("\n3. Creating confusion matrix grid...")
    viz.plot_confusion_matrix_grid(n_samples=16)
    
    print("\n4. Plotting random samples...")
    pred_files = sorted([f for f in os.listdir(viz.pred_dir) if f.endswith('.h5')])
    random_samples = np.random.choice(pred_files, min(10, len(pred_files)), replace=False)
    
    for pred_file in random_samples:
        filename = pred_file.replace('mask', 'image')
        viz.plot_single_prediction(filename, save=True)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"All plots saved to: {viz.output_dir}/")
    print("\nGenerated:")
    print("  ✓ performance_distribution.png")
    print("  ✓ confusion_matrix_grid.png")
    print("  ✓ Best/worst case predictions")
    print("  ✓ Random sample predictions")
    print("="*80)


if __name__ == '__main__':
    main()