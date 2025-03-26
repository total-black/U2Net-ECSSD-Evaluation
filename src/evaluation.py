import os
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, jaccard_score

# Directories
gt_dir = '../data/ECSSD/ground_truth_mask/'
pred_dir = '../results/ECSSD_results/'

gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.png')])
pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])

# Metrics storage
ious = []
dice_scores = []
f1_scores = []

for gt_file, pred_file in zip(gt_files, pred_files):
    gt_path = os.path.join(gt_dir, gt_file)
    pred_path = os.path.join(pred_dir, pred_file)

    gt = np.array(Image.open(gt_path).convert('L'))  # convert to grayscale
    pred = np.array(Image.open(pred_path).convert('L'))

    # Threshold to binary (0 or 1)
    gt = (gt > 128).astype(np.uint8)
    pred = (pred > 128).astype(np.uint8)

    # Flatten for metric computation
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()

    # IoU
    iou = jaccard_score(gt_flat, pred_flat)
    ious.append(iou)

    # Dice Score (F1 for segmentation)
    intersection = np.sum(gt_flat * pred_flat)
    dice = (2. * intersection) / (np.sum(gt_flat) + np.sum(pred_flat) + 1e-7)
    dice_scores.append(dice)

    # F1 Score (binary)
    f1 = f1_score(gt_flat, pred_flat)
    f1_scores.append(f1)

print(f"Mean IoU: {np.mean(ious):.4f}")
print(f"Mean Dice Score: {np.mean(dice_scores):.4f}")
print(f"Mean F1 Score: {np.mean(f1_scores):.4f}")
