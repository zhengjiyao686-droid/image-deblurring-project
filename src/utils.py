import os
import shutil
import random
import cv2

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def list_images(folder):
    """Return sorted list of image paths in a folder."""
    extensions = ('.jpg', '.jpeg', '.png')
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(extensions)
    ])

def split_dataset(source_blur, source_sharp,
                  out_dir, train=0.7, val=0.15, test=0.15,
                  seed=42):
    """
    Split paired blur/sharp images into train/val/test sets.
    """
    assert abs(train + val + test - 1.0) < 1e-6, "Ratios must sum to 1"

    blur_imgs  = list_images(source_blur)
    sharp_imgs = list_images(source_sharp)
    assert len(blur_imgs) == len(sharp_imgs), \
        "Blur and sharp folders must have the same number of images"

    pairs = list(zip(blur_imgs, sharp_imgs))
    random.seed(seed)
    random.shuffle(pairs)

    n       = len(pairs)
    n_train = int(n * train)
    n_val   = int(n * val)

    splits = {
        'train': pairs[:n_train],
        'val':   pairs[n_train:n_train + n_val],
        'test':  pairs[n_train + n_val:]
    }

    for split, pair_list in splits.items():
        for subfolder in ('blur', 'sharp'):
            ensure_dir(os.path.join(out_dir, split, subfolder))
        for blur_p, sharp_p in pair_list:
            shutil.copy(blur_p,
                os.path.join(out_dir, split, 'blur',
                             os.path.basename(blur_p)))
            shutil.copy(sharp_p,
                os.path.join(out_dir, split, 'sharp',
                             os.path.basename(sharp_p)))

    for split, pair_list in splits.items():
        print(f"{split}: {len(pair_list)} image pairs")

    return splits

def resize_image(image, width=640):
    """Resize image to given width, keeping aspect ratio."""
    h, w = image.shape[:2]
    ratio = width / w
    return cv2.resize(image, (width, int(h * ratio)))