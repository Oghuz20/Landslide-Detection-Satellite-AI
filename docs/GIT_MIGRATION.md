# Git Migration Guide

> How to move your existing GitHub repository to a branch and push the new clean version to main

---

## Overview

This guide helps you:
1. Save your existing GitHub content to a branch called `old-unet`
2. Replace the main branch with the new, clean project structure
3. Keep your repository history intact

---

## Prerequisites

✅ Completed cleanup: `.\scripts\cleanup_old_files.ps1`  
✅ Completed reorganization: `.\scripts\reorganize_structure.ps1`  
✅ All new files created (README.md, .gitignore, etc.)

---

## Step 1: Initialize Git (if not already done)

If your project is not yet a Git repository:

```powershell
cd D:\Graduation_Work\Landslide-Detection-Satellite-AI
git init
```

---

## Step 2: Save Current State to Branch

### A. Check Current Branch

```powershell
git branch
```

If you see `* main` or `* master`, continue. If not, create the main branch:

```powershell
git checkout -b main
```

### B. Add All Current Files

```powershell
git add .
git commit -m "Save old U-Net project state before reorganization"
```

### C. Create Archive Branch

```powershell
# Create new branch from current state
git branch old-unet

# Verify branch was created
git branch
```

You should now see:
```
  old-unet
* main
```

---

## Step 3: Clean Main Branch

### A. Remove All Files (Keep Git History)

```powershell
# Remove all tracked files
git rm -rf .

# Commit the removal
git commit -m "Clean slate for reorganized project"
```

### B. Add New Clean Structure

```powershell
# Add all new files
git add .

# Check what will be committed
git status

# Commit the new structure
git commit -m "Add clean reorganized project structure

- Attention U-Net architecture
- Proper data splits (no leakage)
- Ensemble prediction method
- Comprehensive documentation
- Professional README and results analysis
"
```

---

## Step 4: Push to GitHub

### A. Add Remote (if not already added)

```powershell
# Check if remote exists
git remote -v

# If empty, add GitHub remote
git remote add origin https://github.com/Oghuz20/Landslide-Detection-Satellite-AI.git
```

### B. Push Both Branches

```powershell
# Push main branch (force push to overwrite)
git push -u origin main --force

# Push archive branch
git push -u origin old-unet
```

**⚠️ Warning**: The `--force` flag will overwrite the current main branch on GitHub. Make sure you've saved the old content to `old-unet` branch first!

---

## Step 5: Verify on GitHub

1. Go to: https://github.com/Oghuz20/Landslide-Detection-Satellite-AI
2. Check that main branch shows the new clean structure
3. Switch to `old-unet` branch to see archived content
4. Verify README looks professional

---

## Step 6: Set Branch Descriptions (Optional)

On GitHub, you can add branch descriptions:

**Main Branch**:
> Clean, production-ready landslide detection with Attention U-Net. F1: 0.5963 on test set.

**Old-UNet Branch**:
> Archive of original U-Net experiments. Kept for reference and historical comparison.

---

## Alternative: Two Separate Repositories

If you prefer to keep them completely separate:

### A. Create New Repository for Clean Version

```powershell
# In your cleaned project directory
git init
git add .
git commit -m "Initial commit: Landslide detection with Attention U-Net"

# Add new remote
git remote add origin https://github.com/Oghuz20/Landslide-Detection-Attention-UNet.git
git push -u origin main
```

### B. Keep Old Repository Unchanged

Your existing repository remains at:
https://github.com/Oghuz20/Landslide-Detection-Satellite-AI

---

## Recommended: Single Repository with Branches

**Advantages**:
- ✅ Maintains history
- ✅ Easy comparison between versions
- ✅ Shows progression of work
- ✅ Single URL for both versions

**Main Branch**: Clean, professional version  
**Old-UNet Branch**: Historical reference

---

## GitHub Repository Settings

After pushing, configure your repository:

### 1. Set Default Branch

- Go to: Settings → Branches
- Set default branch to `main`
- This ensures visitors see the clean version first

### 2. Add Topics

Add relevant topics to help others find your repo:

```
landslide-detection
deep-learning
semantic-segmentation
attention-unet
pytorch
satellite-imagery
sentinel-2
remote-sensing
computer-vision
earth-observation
```

### 3. Add Description

```
🌋 Automated landslide detection from satellite imagery using Attention U-Net deep learning model. Processes 14-band multispectral data (Sentinel-2 + DEM) for accurate landslide segmentation.
```

### 4. Add Website (Optional)

If you create a GitHub Pages site or deployment:
```
https://oghuz20.github.io/Landslide-Detection-Satellite-AI
```

### 5. Enable Issues and Discussions

- ✅ Enable Issues (for bug reports)
- ✅ Enable Discussions (for Q&A)

---

## What's in Each Branch?

### Main Branch (Clean)

```
Landslide-Detection-Satellite-AI/
├── data/                     # (in .gitignore)
├── src/                      # Clean source code
├── checkpoints/              # Best models only
├── predictions/test_final/   # Final results only
├── visualizations/           # Professional plots
├── docs/                     # Comprehensive documentation
├── notebooks/                # Colab training notebook
├── ensemble_predict.py       # Main inference script
├── README.md                 # Professional documentation
├── requirements.txt          # Dependencies
├── .gitignore               # Proper ignores
└── LICENSE                   # MIT License
```

### Old-UNet Branch (Archive)

```
Landslide-Detection-Satellite-AI/
├── checkpoints/              # Old checkpoints
├── checkpoints_full/         # Previous attempts
├── predictions/test/         # Old predictions
├── predictions/valid_full/   # Leaked validation
├── src/train_final.py        # Old training script
├── README_old.md             # Old documentation
└── ...                       # All historical files
```

---

## Updating Your Thesis/Documentation

When referencing your GitHub in your thesis:

**Old Way (Don't Use)**:
```
The code is available at: https://github.com/Oghuz20/Landslide-Detection-Satellite-AI
```

**New Way (Recommended)**:
```
The code is available at: 
https://github.com/Oghuz20/Landslide-Detection-Satellite-AI

Final model (Attention U-Net): main branch
Historical experiments: old-unet branch
```

---

## Troubleshooting

### Problem: "Updates were rejected"

**Solution**: You need to force push (after confirming you've saved old content):

```powershell
git push origin main --force
```

### Problem: "Permission denied"

**Solution**: Set up SSH key or use personal access token:

```powershell
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add to GitHub: Settings → SSH Keys → New SSH key
# Then use SSH URL:
git remote set-url origin git@github.com:Oghuz20/Landslide-Detection-Satellite-AI.git
```

### Problem: Large Files Rejected

**Solution**: Ensure `.gitignore` is working:

```powershell
# Check what's being tracked
git ls-files

# If large files are tracked, remove them:
git rm --cached checkpoints/*.pth
git rm --cached data/TrainData/*
git commit -m "Remove large files from tracking"
```

### Problem: Branch Already Exists

**Solution**: Delete and recreate:

```powershell
# Delete branch
git branch -D old-unet

# Recreate from current state
git branch old-unet
```

---

## Success Checklist

After migration, verify:

- [ ] Main branch shows clean project structure
- [ ] README.md displays properly on GitHub
- [ ] Old-unet branch contains historical content
- [ ] .gitignore prevents large files from being tracked
- [ ] All documentation files are present
- [ ] Repository description and topics are set
- [ ] Default branch is set to `main`

---

## Next Steps

1. ✅ Push both branches to GitHub
2. ✅ Configure repository settings
3. ✅ Create GitHub Pages documentation (optional)
4. ✅ Share repository link in your thesis
5. ✅ Add repository link to your resume/CV

---

**Questions?** Open an issue in the repository or check the documentation in `docs/`.
