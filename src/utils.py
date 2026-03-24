import os

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def list_images(folder):
    """Return list of image paths in a folder."""
    extensions = ('.jpg', '.jpeg', '.png')
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(extensions)
    ]