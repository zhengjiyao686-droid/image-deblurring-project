import cv2
import numpy as np

def apply_wiener_filter(image):
    """Apply Wiener filter for motion deblurring."""
    # TODO: Implement Wiener filter
    pass

def apply_gaussian_blur(image, kernel_size=5):
    """Apply Gaussian blur to simulate motion blur."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def load_image(path):
    """Load an image from disk."""
    return cv2.imread(path)

def save_image(path, image):
    """Save an image to disk."""
    cv2.imwrite(path, image)