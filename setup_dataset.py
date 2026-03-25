
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import split_dataset

GOPRO_BLUR  = r"F:\Courses\semester2\Computer Vision\assignment1\Gopro\train\blur"
GOPRO_SHARP = r"F:\Courses\semester2\Computer Vision\assignment1\Gopro\train\sharp"
OUTPUT_DIR  = "data"

print("Splitting dataset into train / val / test ...")
split_dataset(
    source_blur  = GOPRO_BLUR,
    source_sharp = GOPRO_SHARP,
    out_dir      = OUTPUT_DIR,
    train        = 0.70,
    val          = 0.15,
    test         = 0.15,
    seed         = 42
)
print("Done! Dataset is ready in the data/ folder.")