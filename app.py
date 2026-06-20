from fastapi import FastAPI
from torch.utils.data import DataLoader, random_split
import torch
from src.dataset_ver1 import augmentation_transforms
from src.train_with_DiceLoss import train
from src.test_ver1 import batch_inference

state = {
    "training": False,
    "trained": False,
    "backbone": "mobilenet_v3_large",
    "img_width": 300,
    "img_height": 150
}


def build_dataset(image_dir, mask_dir, transform=None):
    from src.dataset_ver1 import MyDataset
    dataset = MyDataset(image_dir=image_dir, mask_dir=mask_dir, transform= transform)
    return dataset

def build_dataloader(dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
    return dataloader

app = FastAPI( title="DeepLabV3 Training API", description="API for training DeepLabV3 model with Dice Loss", version="1.0.0")


@app.get(
    "/status",
    tags=["Status"],
    summary="Get training status",
)
def status():
    """Return whether the model is trained/training and which device is available."""
    if(model_path := f"./model/best_deeplabv3_{state['backbone']}_segmentation.pth") is not None :
            state["trained"] = True
    return {
        "trained":    state["trained"],
        "training":   state["training"],
        "device":     "cuda" if torch.cuda.is_available() else "cpu",
    }

@app.post(
    "/set_dataset_path",
    tags=["Dataset"],
    summary="Set dataset root path",
    description="""Set the root path for the dataset. The dataset should have two subdirectories: 'images' and 'masks""",
    )
def set_dataset_path(image_dir: str, mask_dir: str):
    """Set the root path for the dataset. The dataset should have two subdirectories: 'images' and 'masks'."""
    state["image_dir"] = image_dir
    state["mask_dir"] = mask_dir
    print(f"Dataset paths set: image_dir={image_dir}, mask_dir={mask_dir}")
    return {"message": "Dataset paths set successfully."}

@app.post(
    "/train",
    tags=["Training"],
    summary="Start training the model",
    description="""Start training the DeepLabV3 model with Dice Loss. Can be used with different backbones (mobilenet_v3_large, resnet50). The dataset paths must be set before calling this endpoint.""",
)
def train_with_dice_loss(backbone: str = "mobilenet_v3_large", epochs: int = 3, num_classes: int = 2, batch_size: int = 2, lr: float = 1e-3, img_width: int = 300, img_height: int = 150):
    """Start training the DeepLabV3 model with Dice Loss. The dataset paths must be set before calling this endpoint."""
    if "image_dir" not in state or "mask_dir" not in state:
        return {"error": "Dataset paths not set. Please set them using /set_dataset_path endpoint."}
    state["training"] = True
    state["backbone"] = backbone
    img_size = (img_height, img_width)
    state["img_width"] = img_width
    state["img_height"] = img_height
    dataset = build_dataset(image_dir=state["image_dir"], mask_dir=state["mask_dir"], transform=augmentation_transforms(img_size=img_size))
    train_dataset, val_dataset = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

    train_dataloader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_dataloader = build_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False) 
    message = train(backbone=backbone, epochs=epochs, num_classes=num_classes, lr=lr, device=None, train_dataloader=train_dataloader, val_dataloader=val_dataloader)
  
    state["training"] = False 
    state["trained"] = True
    return {"message": "Training completed successfully.",
            "details": message}

@app.post(
    "/inference",
    tags=["Inference"],
    summary="Run batch inference on a directory of images",
    description="""Run batch inference using a trained model on all images in a specified directory. The results will be saved in the output directory.""",
)
def run_inference(image_dir: str):
    """Run batch inference using a trained model on all images in a specified directory. The results will be saved in the output directory."""
    if not state["trained"]:
        return {"error": "Model weights not found. Please ensure the model is trained and the weights are saved at the specified path."}
    else:
        img_width: int = state.get("img_width", 300)
        img_height: int = state.get("img_height", 150)
        img_size = (img_height, img_width)
        output_dir: str = "./result"
        num_classes: int = 2
        model_path: str = f"./model/best_deeplabv3_{state['backbone']}_segmentation.pth"
        batch_inference(backbone=state['backbone'], image_dir=image_dir, model_path=model_path, output_dir=output_dir, num_classes=num_classes, img_size=img_size)
    return {"message": f"Inference completed. Results saved to {output_dir}."}

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading
    import time

    def open_docs():
        time.sleep(1)  
        webbrowser.open("http://127.0.0.1:8001/docs")

    threading.Thread(target=open_docs).start()

    uvicorn.run(app, host="127.0.0.1", port=8001)
