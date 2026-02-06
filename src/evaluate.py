"""
Evaluation script for landslide detection model
Computes metrics on validation and test sets
"""
import os
import sys
import h5py
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, 
    precision_score, 
    recall_score, 
    confusion_matrix,
    classification_report
)


def load_predictions(pred_dir):
    """
    Load predictions from directory
    
    Args:
        pred_dir: Directory containing .h5 prediction files
        
    Returns:
        all_preds: Flattened array of all predictions
        all_targets: Flattened array of all ground truth
    """
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.h5')])
    
    all_preds = []
    all_targets = []
    
    print(f"Loading {len(pred_files)} predictions...")
    
    for pred_file in tqdm(pred_files):
        # Load prediction
        pred_path = os.path.join(pred_dir, pred_file)
        with h5py.File(pred_path, 'r') as f:
            pred = f['mask'][:]
        
        # Load ground truth (assume same structure)
        gt_dir = pred_dir.replace('predictions', 'data').replace('test_attention_ensemble', 'TestData/mask')
        gt_path = os.path.join(gt_dir, pred_file)
        
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth not found for {pred_file}")
            continue
            
        with h5py.File(gt_path, 'r') as f:
            target = f['mask'][:]
        
        # Flatten and append
        all_preds.extend(pred.flatten())
        all_targets.extend(target.flatten())
    
    return np.array(all_preds), np.array(all_targets)


def compute_metrics(y_true, y_pred):
    """
    Compute comprehensive metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        dict of metrics
    """
    # Basic metrics
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    accuracy = (y_true == y_pred).mean()
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'confusion_matrix': cm
    }


def print_metrics(metrics, dataset_name='Test'):
    """
    Pretty print metrics
    
    Args:
        metrics: Dictionary of computed metrics
        dataset_name: Name of dataset being evaluated
    """
    print("\n" + "="*80)
    print(f"{dataset_name.upper()} SET RESULTS")
    print("="*80)
    
    print(f"\nF1 Score:   {metrics['f1']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    
    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:>10,}  |  FP: {cm[0,1]:>10,}")
    print(f"  FN: {cm[1,0]:>10,}  |  TP: {cm[1,1]:>10,}")
    
    # Derived metrics
    if cm[0,1] + cm[1,1] > 0:  # Avoid division by zero
        precision_manual = cm[1,1] / (cm[0,1] + cm[1,1])
        print(f"\nPositive Predictive Value: {precision_manual:.4f}")
    
    if cm[1,0] + cm[1,1] > 0:
        recall_manual = cm[1,1] / (cm[1,0] + cm[1,1])
        print(f"Sensitivity (True Positive Rate): {recall_manual:.4f}")
    
    if cm[0,0] + cm[0,1] > 0:
        specificity = cm[0,0] / (cm[0,0] + cm[0,1])
        print(f"Specificity (True Negative Rate): {specificity:.4f}")
    
    print("="*80)


def main():
    """
    Main evaluation function
    """
    # Configuration
    PRED_DIR = './predictions/test_attention_ensemble'
    
    if not os.path.exists(PRED_DIR):
        print(f"Error: Prediction directory not found: {PRED_DIR}")
        return
    
    print("="*80)
    print("LANDSLIDE DETECTION - MODEL EVALUATION")
    print("="*80)
    print(f"\nPredictions: {PRED_DIR}")
    
    # Load predictions
    y_pred, y_true = load_predictions(PRED_DIR)
    
    print(f"\nTotal pixels evaluated: {len(y_true):,}")
    print(f"Landslide pixels (ground truth): {y_true.sum():,} ({y_true.mean()*100:.2f}%)")
    print(f"Landslide pixels (predicted): {y_pred.sum():,} ({y_pred.mean()*100:.2f}%)")
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    
    # Print results
    print_metrics(metrics, dataset_name='Test')
    
    # Save to file
    output_file = 'evaluation_results.txt'
    with open(output_file, 'w') as f:
        f.write("LANDSLIDE DETECTION - EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"F1 Score:   {metrics['f1']:.4f}\n")
        f.write(f"Precision:  {metrics['precision']:.4f}\n")
        f.write(f"Recall:     {metrics['recall']:.4f}\n")
        f.write(f"Accuracy:   {metrics['accuracy']:.4f}\n\n")
        
        cm = metrics['confusion_matrix']
        f.write("Confusion Matrix:\n")
        f.write(f"  TN: {cm[0,0]:>10,}  |  FP: {cm[0,1]:>10,}\n")
        f.write(f"  FN: {cm[1,0]:>10,}  |  TP: {cm[1,1]:>10,}\n")
    
    print(f"\n✓ Results saved to {output_file}")


if __name__ == '__main__':
    main()