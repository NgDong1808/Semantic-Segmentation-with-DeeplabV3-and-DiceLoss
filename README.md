# DeepLabV3 Crack Segmentation

A simple PyTorch-based semantic segmentation project using DeepLabV3 to detect cracks in pill/ginseng images.

## Project Structure

- `src/` - source code for dataset loading, training, and testing
- `data/` - dataset folders (excluded from git)
- `model/` - saved model weights (excluded from git)
- `result/` - inference outputs (excluded from git)
- `requirements.txt` - Python dependencies
- `.gitignore` - ignored files and folders
- `LICENSE` - open-source license

## Example Output

The example below shows the predicted overlay output for `predicted_pill_ginseng_crack_007.png`.

![Predicted crack overlay](predicted_pill_ginseng_crack_007.png)

## Getting Started

### 1. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare your data

Place your test images into a folder such as:

```text
data/test/images/crack/
```

If you also have training images, follow the existing data layout under `data/`.

### 4. Run inference

Use the `batch_inference` function from `src/test_ver1.py`.

```python
from src.test_ver1 import batch_inference

batch_inference(
    backbone='resnet50',
    image_dir='data/test/images/crack',
    model_path='model/best_deeplabv3_crack_segmentation.pth',
    output_dir='result',
    num_classes=2,
    img_size=(150, 300)
)
```

This will process all supported images in `image_dir` and save overlay results into `result/`.

## Training Notes

The training script is located at `src/train_with_DiceLoss.py`.
This project uses DeepLabV3 with a custom classifier head configured for `num_classes=2`.

To train the model:

1. Prepare image and mask datasets.
2. Create training and validation dataloaders.
3. Call `train(...)` from `src/train_with_DiceLoss.py`.

## Recommended Git Ignore Rules

The repository ignores:

- `data/`
- `model/`
- `result/`
- `*.pth`, `*.pt`
- virtual environment folders like `.venv/`

## License

This project is licensed under the MIT License. See `LICENSE`.
