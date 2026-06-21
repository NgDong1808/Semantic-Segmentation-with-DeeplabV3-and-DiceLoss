import torch
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50
from torchgen import model
def export_model_to_onnx(backbone, model_path, onnx_output_path, num_classes, img_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if backbone == "mobilenet_v3_large":
        model = deeplabv3_mobilenet_v3_large(weights=None, aux_loss=True)
    elif backbone == "resnet50":
        model = deeplabv3_resnet50(weights=None, aux_loss=True)
    else:
        raise ValueError("Unsupported backbone. Please choose 'mobilenet_v3_large' or 'resnet50'.")
    
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


    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1], device=device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        opset_version=17,
        input_names=["image"],
        output_names=["mask"],
        dynamic_axes={
            "image": {
                0: "batch",
                2: "height",
                3: "width"
            },
            "mask": {
                0: "batch",
                2: "height",
                3: "width"
            }
        }
    )