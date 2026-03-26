# Object detection module using pretrained Faster R-CNN (COCO)

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os
from torchvision import models, transforms
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights
)

# COCO class names (80 classes)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle',
    'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

# Model loader

def load_model(device=None):
    """Load pretrained Faster R-CNN (COCO) and move to device."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Faster R-CNN on: {device}")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model   = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    return model, device

# Inference

def run_detection(image_bgr, model, device, threshold=0.5):
    """
    Run Faster R-CNN on a single BGR image.
    Returns dict with boxes, labels, scores and latency.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor    = transform(image_rgb).unsqueeze(0).to(device)

    start = time.time()
    with torch.no_grad():
        outputs = model(tensor)
    latency = time.time() - start

    boxes   = outputs[0]['boxes'].cpu().numpy()
    labels  = outputs[0]['labels'].cpu().numpy()
    scores  = outputs[0]['scores'].cpu().numpy()

    # Filter by confidence threshold
    mask    = scores >= threshold
    return {
        'boxes':   boxes[mask],
        'labels':  labels[mask],
        'scores':  scores[mask],
        'latency': latency,
        'all_scores': scores  # keep all for PR curve
    }

def run_detection_batch(image_paths, model, device,
                        threshold=0.5, max_size=640):
    """
    Run detection on a list of image paths.
    Resizes images to max_size for speed.
    Returns list of result dicts.
    """
    results = []
    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        # Resize for speed
        h, w  = img.shape[:2]
        scale = max_size / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
        result = run_detection(img, model, device, threshold)
        result['path'] = path
        results.append(result)
        if (i+1) % 10 == 0:
            print(f"  Processed {i+1}/{len(image_paths)} images")
    return results

# Metrics

def compute_map_from_results(results_list, iou_threshold=0.5):
    """
    Compute pseudo-mAP from detection results.
    Since GoPro has no GT boxes, we use sharp-image detections
    as pseudo ground truth and compare blurry/deblurred against them.
    """
    all_scores  = []
    all_labels  = []
    det_counts  = []

    for r in results_list:
        all_scores.extend(r['scores'].tolist())
        all_labels.extend(r['labels'].tolist())
        det_counts.append(len(r['scores']))

    return {
        'mean_confidence': float(np.mean(all_scores)) if all_scores else 0.0,
        'total_detections': sum(det_counts),
        'avg_detections_per_image': float(np.mean(det_counts)) if det_counts else 0.0,
        'per_class_counts': _per_class_counts(all_labels),
        'per_class_confidence': _per_class_confidence(all_labels, all_scores)
    }

def _per_class_counts(labels):
    counts = {}
    for l in labels:
        name = COCO_CLASSES[l] if l < len(COCO_CLASSES) else str(l)
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))

def _per_class_confidence(labels, scores):
    conf = {}
    for l, s in zip(labels, scores):
        name = COCO_CLASSES[l] if l < len(COCO_CLASSES) else str(l)
        conf.setdefault(name, []).append(s)
    return {k: float(np.mean(v)) for k, v in conf.items()}

def compute_precision_recall(results_sharp, results_query,
                             score_thresholds=None):
    """
    Compute precision-recall curve.
    Uses sharp detections as proxy ground truth.
    """
    if score_thresholds is None:
        score_thresholds = np.linspace(0.3, 0.95, 20)

    precisions, recalls = [], []

    for thresh in score_thresholds:
        tp = fp = fn = 0
        for r_s, r_q in zip(results_sharp, results_query):
            gt_count  = int(np.sum(r_s['scores'] >= thresh))
            det_count = int(np.sum(r_q['scores'] >= thresh))
            matched   = min(gt_count, det_count)
            tp += matched
            fp += max(0, det_count - matched)
            fn += max(0, gt_count  - matched)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)

    return np.array(recalls), np.array(precisions)

# Visualisation

def draw_detections(image_bgr, result, title='',
                    threshold=0.5, max_boxes=15):
    """Draw bounding boxes and labels on image."""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.imshow(img_rgb)

    colors = plt.cm.Set3(np.linspace(0, 1, len(COCO_CLASSES)))

    for i, (box, label, score) in enumerate(
            zip(result['boxes'], result['labels'], result['scores'])):
        if i >= max_boxes:
            break
        x1, y1, x2, y2 = box
        w, h   = x2 - x1, y2 - y1
        color  = colors[label % len(colors)]
        rect   = patches.Rectangle((x1, y1), w, h,
                                    linewidth=2, edgecolor=color,
                                    facecolor='none')
        ax.add_patch(rect)
        name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else str(label)
        ax.text(x1, y1 - 4, f'{name} {score:.2f}',
                fontsize=8, color='white',
                bbox=dict(facecolor=color, alpha=0.7, pad=1))

    n = len(result['scores'])
    ax.set_title(f'{title}  |  {n} detections  '
                 f'| avg conf: {np.mean(result["scores"]):.3f}'
                 if n > 0 else f'{title}  |  0 detections',
                 fontsize=11)
    ax.axis('off')
    plt.tight_layout()
    return fig

def plot_confidence_distribution(results_blur, results_deblur,
                                  results_sharp=None,
                                  save_path=None):
    """Histogram of confidence scores across methods."""
    fig, ax = plt.subplots(figsize=(10, 5))

    def get_scores(results):
        s = []
        for r in results:
            s.extend(r['scores'].tolist())
        return s

    ax.hist(get_scores(results_blur),   bins=30, alpha=0.6,
            label='Blurry',      color='#E24B4A')
    ax.hist(get_scores(results_deblur), bins=30, alpha=0.6,
            label='Deblurred',   color='#1D9E75')
    if results_sharp:
        ax.hist(get_scores(results_sharp), bins=30, alpha=0.6,
                label='Sharp (GT)', color='#534AB7')

    ax.set_xlabel('Confidence score', fontsize=12)
    ax.set_ylabel('Count',            fontsize=12)
    ax.set_title('Confidence score distribution by image type',
                 fontsize=13)
    ax.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig

def plot_precision_recall_curves(results_sharp,
                                  results_blur,
                                  results_deblur,
                                  save_path=None):
    """Plot PR curves for blurry vs deblurred vs sharp."""
    thresholds = np.linspace(0.3, 0.95, 20)

    r_blur,    p_blur    = compute_precision_recall(
        results_sharp, results_blur,    thresholds)
    r_deblur,  p_deblur  = compute_precision_recall(
        results_sharp, results_deblur,  thresholds)
    r_sharp,   p_sharp   = compute_precision_recall(
        results_sharp, results_sharp,   thresholds)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(r_blur,   p_blur,   'o-', color='#E24B4A',
            label='Blurry',      linewidth=2)
    ax.plot(r_deblur, p_deblur, 's-', color='#1D9E75',
            label='Deblurred',   linewidth=2)
    ax.plot(r_sharp,  p_sharp,  '^-', color='#534AB7',
            label='Sharp (GT)',  linewidth=2)

    ax.set_xlabel('Recall',    fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall curves: blurry vs deblurred vs sharp',
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig

def plot_per_class_ap(results_blur, results_deblur,
                      results_sharp=None, top_n=10,
                      save_path=None):
    """Bar chart of per-class average confidence (proxy for AP)."""
    def class_conf(results):
        labels, scores = [], []
        for r in results:
            labels.extend(r['labels'].tolist())
            scores.extend(r['scores'].tolist())
        return _per_class_confidence(labels, scores)

    conf_blur    = class_conf(results_blur)
    conf_deblur  = class_conf(results_deblur)

    # Top N classes by deblurred confidence
    top_classes = sorted(conf_deblur, key=conf_deblur.get,
                         reverse=True)[:top_n]

    x     = np.arange(len(top_classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2,
           [conf_blur.get(c, 0)   for c in top_classes],
           width, label='Blurry',    color='#E24B4A', alpha=0.8)
    ax.bar(x + width/2,
           [conf_deblur.get(c, 0) for c in top_classes],
           width, label='Deblurred', color='#1D9E75', alpha=0.8)

    if results_sharp:
        conf_sharp = class_conf(results_sharp)
        ax.bar(x, [conf_sharp.get(c, 0) for c in top_classes],
               0.1, label='Sharp', color='#534AB7', alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(top_classes, rotation=35, ha='right',
                       fontsize=10)
    ax.set_ylabel('Avg confidence score', fontsize=12)
    ax.set_title(f'Per-class average confidence — top {top_n} classes',
                 fontsize=13)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig

def plot_latency_comparison(lat_blur, lat_deblur,
                             lat_sharp=None, save_path=None):
    """Bar chart of average inference latency per method."""
    labels = ['Blurry', 'Deblurred']
    values = [np.mean(lat_blur), np.mean(lat_deblur)]
    colors = ['#E24B4A', '#1D9E75']

    if lat_sharp:
        labels.append('Sharp')
        values.append(np.mean(lat_sharp))
        colors.append('#534AB7')

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.4)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f'{val:.3f}s', ha='center', fontsize=11)

    ax.set_ylabel('Avg inference time (s)', fontsize=12)
    ax.set_title('Detection latency comparison', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
    return fig