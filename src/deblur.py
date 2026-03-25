# Motion deblurring module — classical and deep learning methods

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os

# Classical Method 1: Wiener Filter

def wiener_filter(image, kernel_size=5, noise_var=0.01):
    """Apply Wiener filter for motion deblurring."""
    if len(image.shape) == 3:
        channels = cv2.split(image)
        filtered = [_wiener_single(c, kernel_size, noise_var) for c in channels]
        return cv2.merge(filtered)
    return _wiener_single(image, kernel_size, noise_var)

def _wiener_single(channel, kernel_size, noise_var):
    """Wiener filter on a single grayscale channel."""
    img_float  = channel.astype(np.float64) / 255.0
    kernel     = np.ones((kernel_size, kernel_size), np.float64) / (kernel_size ** 2)
    kernel_fft = np.fft.fft2(kernel, s=img_float.shape)
    img_fft    = np.fft.fft2(img_float)
    kernel_pwr = np.abs(kernel_fft) ** 2
    wiener_fft = (np.conj(kernel_fft) / (kernel_pwr + noise_var)) * img_fft
    result     = np.abs(np.fft.ifft2(wiener_fft))
    result     = np.clip(result, 0, 1)
    return (result * 255).astype(np.uint8)

# Classical Method 2: Richardson-Lucy

def richardson_lucy(image, kernel_size=5, iterations=30):
    """Richardson-Lucy deconvolution."""
    if len(image.shape) == 3:
        channels  = cv2.split(image)
        deblurred = [_rl_single(c, kernel_size, iterations) for c in channels]
        return cv2.merge(deblurred)
    return _rl_single(image, kernel_size, iterations)

def _rl_single(channel, kernel_size, iterations):
    """Richardson-Lucy on a single channel."""
    img      = channel.astype(np.float64) / 255.0
    kernel   = np.ones((kernel_size, kernel_size), np.float64) / (kernel_size ** 2)
    estimate = img.copy()
    for _ in range(iterations):
        blurred  = cv2.filter2D(estimate, -1, kernel)
        blurred  = np.where(blurred == 0, 1e-6, blurred)
        ratio    = img / blurred
        estimate = estimate * cv2.filter2D(ratio, -1, np.flip(kernel))
        estimate = np.clip(estimate, 0, 1)
    return (estimate * 255).astype(np.uint8)

# Deep Learning Method: DnCNN-style

def get_dncnn_model():
    """Build a lightweight DnCNN-style model."""
    try:
        import torch
        import torch.nn as nn

        class DnCNN(nn.Module):
            def __init__(self, depth=17, n_channels=64, image_channels=3):
                super().__init__()
                layers = [
                    nn.Conv2d(image_channels, n_channels, 3, padding=1),
                    nn.ReLU(inplace=True)
                ]
                for _ in range(depth - 2):
                    layers += [
                        nn.Conv2d(n_channels, n_channels, 3, padding=1, bias=False),
                        nn.BatchNorm2d(n_channels),
                        nn.ReLU(inplace=True)
                    ]
                layers.append(nn.Conv2d(n_channels, image_channels, 3, padding=1))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return x - self.net(x)

        return DnCNN()

    except ImportError:
        print("PyTorch not found — deep learning method unavailable.")
        return None

def deep_deblur(image, model=None):
    """Apply DnCNN deblurring or unsharp mask fallback."""
    if model is None:
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    try:
        import torch
        model.eval()
        img_float = image.astype(np.float32) / 255.0
        tensor    = torch.from_numpy(
                        img_float.transpose(2, 0, 1)).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor).squeeze(0).numpy()
        output = np.clip(output.transpose(1, 2, 0), 0, 1)
        return (output * 255).astype(np.uint8)
    except Exception as e:
        print(f"Deep deblur error: {e}")
        return image

# Metrics

def compute_psnr(original, restored):
    """Compute PSNR between two images."""
    return psnr(original.astype(np.float64),
                restored.astype(np.float64),
                data_range=255)

def compute_ssim(original, restored):
    """Compute SSIM between two images."""
    if len(original.shape) == 3:
        return ssim(original, restored,
                    channel_axis=2, data_range=255)
    return ssim(original, restored, data_range=255)

# Visualisation

def compare_results(blurry, wiener, rl, deep,
                    sharp=None, save_path=None):
    """Side-by-side comparison plot with PSNR/SSIM metrics."""
    images = [blurry, wiener, rl, deep]
    titles = ['Blurry input', 'Wiener filter',
              'Richardson-Lucy', 'Deep learning']

    if sharp is not None:
        images.insert(0, sharp)
        titles.insert(0, 'Ground truth (sharp)')

    n   = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if sharp is not None and title != 'Ground truth (sharp)':
            p = compute_psnr(sharp, img)
            s = compute_ssim(sharp, img)
            ax.set_title(f'{title}\nPSNR: {p:.2f} dB\nSSIM: {s:.4f}',
                         fontsize=9)
        else:
            ax.set_title(title, fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

# I/O helpers

def load_image(path):
    """Load an image from disk (BGR)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img

def save_image(path, image):
    """Save an image to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)