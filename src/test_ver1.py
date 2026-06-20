import os
import glob
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50
from PIL import Image
from tqdm import tqdm

def batch_inference(backbone: str, image_dir: str, model_path: str, output_dir: str = "./result", num_classes: int = 2, img_size: tuple = (150, 300)):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for inference: {device}")
    
    # Create output folder
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load model
    if backbone == "mobilenet_v3_large":
        model = deeplabv3_mobilenet_v3_large(weights=None, aux_loss=True)
    elif backbone == "resnet50":
        model = deeplabv3_resnet50(weights=None, aux_loss=True)
    else:
        raise ValueError("Invalid backbone. Please choose 'mobilenet_v3_large' or 'resnet50'.")

    # Main classifier
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = torch.nn.Conv2d(
        in_channels,
        num_classes,
        kernel_size=1
    )

    # Aux classifier
    if model.aux_classifier is not None:
        in_channels_aux = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = torch.nn.Conv2d(
            in_channels_aux,
            num_classes,
            kernel_size=1
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at: {model_path}. Please check the path and try again."
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    image_transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Collect images
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.bmp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

    if len(image_paths) == 0:
        print(f"No images found in: {image_dir}")
        return

    print(f"Found {len(image_paths)} images to test.")

    # Class color map (0: background, 1: class 1)
    color_map = {
        0: [0, 0, 0],        
        1: [255, 0, 0]      
    }

    # Inference loop
    for img_path in tqdm(image_paths, desc="Processing Images", colour="cyan"):
        img_name = os.path.basename(img_path)

        original_image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = original_image.size

        # Preprocess image(resize + tensor + normalize)
        image_tensor = image_transform(original_image)   # shape: [3, H_train, W_train]
        image_tensor = image_tensor.unsqueeze(0).to(device)  # [1, 3, H_train, W_train]

        # Inference
        with torch.no_grad():
            outputs = model(image_tensor)["out"]          # [1, num_classes, H_train, W_train]
            preds = torch.argmax(outputs, dim=1).squeeze(0)  # [H_train, W_train]
            preds = preds.cpu().numpy().astype(np.uint8)

        # Resize mask to original image size
        mask_pil = Image.fromarray(preds)
        mask_pil = mask_pil.resize((orig_w, orig_h), resample=Image.NEAREST)
        final_mask = np.array(mask_pil)

        # Create a colored mask based on the class predictions
        colored_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            colored_mask[final_mask == class_id] = color

        original_np = np.array(original_image)
        result = original_np.copy()
        mask_region = final_mask == 1
        red = np.array([255, 0, 0], dtype=np.uint8)
        result[mask_region] = (0.7 * original_np[mask_region] + 0.3 * red).astype(np.uint8)
        # Overlay the colored mask on the original image
        overlay_image = Image.fromarray(result)
     
        # Combine Original | Overlay
        combined = Image.new(
            "RGB",
            (original_image.width * 2, original_image.height)
        )

        combined.paste(original_image, (0, 0))
        combined.paste(overlay_image, (original_image.width, 0))

        # Save
        base_name = os.path.splitext(img_name)[0]
        save_path = os.path.join(output_dir, f"predicted_{base_name}.png")
        combined.save(save_path)

    print(f"Inference completed. Results saved to: {output_dir}")
