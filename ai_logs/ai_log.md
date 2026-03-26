# AI Usage Log
_This file records all AI-assisted code and documentation._

---

## Entry 001 — 2026-03-24

**Tool:** Claude (Anthropic)

**Prompt:**
"Help me correct the incorrect commands in the terminal and design a reasonable structure for my folder."

**Output summary:**
Claude provided step-by-step guidance including folder structure,
README template, .gitignore, requirements.txt, and Git commands.

**Code used:** Yes — .gitignore, README.md, and requirements.txt templates adopted.

**Ethical note:**
Code reviewed manually. No licensing issues. Dataset is publicly
available under Kaggle terms of use.

---

## Entry 002 — 2026-03-24

**Tool:** Claude (Anthropic)

**Prompt:**
"Help me split dataset and build the code framework as required — image deblurring 
with Wiener filter, Richardson-Lucy, and DnCNN methods with PSNR/SSIM evaluation, but
intead of giving me direct functional code, guide me to complete it."

**Output summary:**
Claude provided code framework with all three methods, as well as cpmlete code with
dataset split script, and corrects errors in the code.

**Code used:** Yes — deblur.py, utils.py, setup_dataset.py, project.ipynb

**Ethical note:**
All code reviewed and tested. GoPro dataset used under Kaggle
public terms. DnCNN architecture is a standard published model.

---

## Entry 003 — 2026-03-24

**Tool:** Claude (Anthropic, claude.ai)

**Prompt:**
"Help me create a dataset split script for the GoPro dataset
(70/15/15 train/val/test split) and fix the utils.py module."

**Output summary:**
Claude provided src/utils.py with split_dataset function using
random shuffling with fixed seed=42, and setup_dataset.py script.

**Code adopted:**
- src/utils.py (split_dataset, list_images, resize_image)
- setup_dataset.py

**Review notes:**
Verified split counts: 1472/315/316. Confirmed seed=42 produces
deterministic splits. Tested on actual GoPro path.

**Ethical note:**
No issues. Dataset used under Kaggle academic terms.

---

## Entry 004 — 2026-03-25

**Tool:** Claude (Anthropic, claude.ai)

**Prompt:**
"Guide me to implement Task 3 with Faster R-CNN, but don't give me all the code directly.
Instead, help me step by step to complete the implementation."

**Output summary:**
Claude provided guidance and code with Faster R-CNN inference,
batch detection, precision-recall computation, confidence distribution
plots, per-class AP bar chart, latency comparison, and failure case
analysis.

**Code adopted:**
- src/detect.py (full implementation)
- project.ipynb cells 7-13

**Review notes:**
Verified CUDA detection works on RTX 4060. Tested threshold=0.5 on
50 test images. Confirmed PR curve logic and per-class confidence
computation. Results match expected ranges for COCO pretrained model.

**Ethical note:**
Faster R-CNN pretrained weights from torchvision (BSD License).
COCO class names used for display only, not redistributed.

---

## Entry 005 — 2026-03-25

**Tool:** Claude (Anthropic, claude.ai)

**Prompt:**
"Help me implement Task 4 — pseudo-annotation generation, dataset
construction, and 5-epoch fine-tuning of Faster R-CNN on deblurred images."

**Output summary:**
Claude provided src/train.py with DeblurDetectionDataset, pseudo-annotation
generator (threshold=0.7), build_finetuned_model, train_model with SGD
and StepLR scheduler, training log JSON export, and plot_training_log.
Also guided the completion of project.ipynb cells 14-18.

**Code adopted:**
- src/train.py (full implementation)
- project.ipynb cells 14-18

**Review notes:**
Verified 175/200 images have detections at threshold 0.7.
Training completes in 174s on RTX 4060. Loss curve shows
expected rapid convergence then plateau. Checkpoint saved correctly.

**Ethical note:**
Fine-tuning on pseudo-labels introduces known label noise —
explicitly documented in report as a limitation.

---

