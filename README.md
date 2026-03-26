# Image Deblurring & Object Detection Project

## Overview
Motion deblurring pipeline using GoPro dataset, combining
classical filtering, CNN restoration, and object detection.

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/zhengjiyao686-droid/image-deblurring-project.git
cd image-deblurring-project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
- Visit: https://www.kaggle.com/datasets/rahulbhalley/gopro-deblur
- Place images into: data/train, data/val, data/test

### 4. Prepare the dataset
Edit `setup_dataset.py` to set your GoPro path, then run:
```bash
python setup_dataset.py
```
Expected output:
```
train: 1472 image pairs
val:   315  image pairs
test:  316  image pairs
Done! Dataset is ready in the data/ folder.
```
### 5. Run the notebook
```bash
jupyter notebook project.ipynb
```

## Usage Examples

### Deblur a single image
```python
import sys
sys.path.insert(0, 'src')
from deblur import load_image, richardson_lucy, save_image

img = load_image("data/test/blur/1.png")
result = richardson_lucy(img, kernel_size=5, iterations=30)
save_image("outputs/my_deblurred.png", result)
```

### Run object detection
```python
from detect import load_model, run_detection
import cv2

model, device = load_model()
img = cv2.imread("data/test/blur/1.png")
result = run_detection(img, model, device, threshold=0.5)
print(f"Detections: {len(result['scores'])}")
print(f"Classes: {result['labels']}")
```

### Evaluate PSNR & SSIM
```python
from deblur import load_image, richardson_lucy, compute_psnr, compute_ssim

blur  = load_image("data/test/blur/1.png")
sharp = load_image("data/test/sharp/1.png")
deblurred = richardson_lucy(blur)

print(f"PSNR: {compute_psnr(sharp, deblurred):.2f} dB")
print(f"SSIM: {compute_ssim(sharp, deblurred):.4f}")
```

### Compare all deblurring methods
```python
from deblur import wiener_filter, richardson_lucy, deep_deblur, compare_results

blur  = load_image("data/test/blur/1.png")
sharp = load_image("data/test/sharp/1.png")

compare_results(
    blurry = blur,
    wiener = wiener_filter(blur),
    rl     = richardson_lucy(blur),
    deep   = deep_deblur(blur),
    sharp  = sharp,
    save_path = "outputs/my_comparison.png"
)
```

Run cells in order to reproduce all results.
## AI Tools Used
All AI prompts and outputs are logged in /ai_logs/ai_log.md

**Tool used:** Claude (Anthropic, claude.ai) and GitHub Copilot (copilot.github.com)

## Ethical Considerations
- Dataset used under Kaggle public terms
- All AI-generated code is reviewed before use
- No personal or sensitive data is processed

## Requirements
```
numpy
opencv-python
matplotlib
torch
torchvision
ultralytics
jupyter
Pillow
scikit-image
scipy
tqdm
```

## Repository

**GitHub:** https://github.com/zhengjiyao686-droid/image-deblurring-project
**Student:** Jiyao Zheng | a1952160@adelaide.edu.au
**Course:** COMP6001 Computer Vision | University of Adelaide | 2026