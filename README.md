# DeepLabV3 Crack Segmentation

Một project ví dụ dùng DeepLabV3 (PyTorch) để phát hiện vết nứt trên ảnh.

## Nội dung
- `src/` - mã nguồn (dataset, train, test)
- `data/` - dữ liệu (không commit dữ liệu lớn)
- `model/` - weights đã huấn luyện (không commit)
- `result/` - kết quả inference (không commit)

## Nhanh chóng bắt đầu
1. Tạo virtual environment và cài dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Unix / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

2. Huấn luyện (ví dụ):

```python
# Tùy biến theo script train của bạn
from src.train_with_DiceLoss import train
# chỉnh path, dataloaders trước khi gọi
# train(epochs=10, ...)
```

3. Chạy inference (ví dụ):

```python
from src.test_ver1 import batch_inference
batch_inference(backbone='resnet50', image_dir='data/test/images/crack', model_path='model/best_deeplabv3_crack_segmentation.pth', output_dir='result')
```

## Lưu ý khi push lên GitHub
- Không commit dữ liệu lớn, weights hoặc kết quả: `data/`, `model/`, `result/` đã được thêm vào `.gitignore`.

## License
Project này được cấp phép dưới MIT License — xem file `LICENSE`.
