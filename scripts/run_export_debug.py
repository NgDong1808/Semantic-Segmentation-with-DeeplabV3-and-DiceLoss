import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.export_model import export_model_to_onnx


if __name__ == "__main__":
    export_model_to_onnx(
        backbone="mobilenet_v3_large",
        model_path="model/best_deeplabv3_mobilenet_v3_large_segmentation.pth",
        num_classes=2,
        img_size=(512, 512)
    )
