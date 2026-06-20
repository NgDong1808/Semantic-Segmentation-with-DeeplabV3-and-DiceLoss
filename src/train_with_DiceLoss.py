import torch
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50
from tqdm.autonotebook import tqdm
import segmentation_models_pytorch as smp
from torch import nn
import numpy as np
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassJaccardIndex, MulticlassF1Score
import os

def train(backbone: str = "mobilenet_v3_large", epochs: int = 3, num_classes: int = 2, lr: float = 1e-3, device=None, train_dataloader=None, val_dataloader=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if backbone == "mobilenet_v3_large":
        model = deeplabv3_mobilenet_v3_large(weights="DEFAULT")
    elif backbone == "resnet50":
        model = deeplabv3_resnet50(weights="DEFAULT")
    else:
        raise ValueError("Invalid backbone. Please choose 'mobilenet_v3_large' or 'resnet50'.")

    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    if model.aux_classifier is not None:
        in_channels_aux = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(in_channels_aux, num_classes, kernel_size=1)

    # Gen metrics
    metric_precision = MulticlassPrecision(num_classes=num_classes, average=None, ignore_index=255).to(device)
    metric_recall = MulticlassRecall(num_classes=num_classes, average=None, ignore_index=255).to(device)
    metric_f1 = MulticlassF1Score(num_classes=num_classes, average=None, ignore_index=255).to(device)
    metric_iou = MulticlassJaccardIndex(num_classes=num_classes, average=None, ignore_index=255).to(device)
    model.to(device)

    # Loss
    criterion = smp.losses.DiceLoss(mode='multiclass', ignore_index=255)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) 
    
    best_miou = 0.0
    best_epoch = 0
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = []
        progress_bar = tqdm(train_dataloader, colour="green", desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)
      
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs['out'], masks)
            
            if 'aux' in outputs and outputs['aux'] is not None:
                loss_aux = criterion(outputs['aux'], masks)
                loss = loss + 0.4 * loss_aux          

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            progress_bar.set_postfix(loss= np.mean(train_loss))
        print(f"Epoch {epoch+1} - Train Loss Trung bình: {np.mean(train_loss):.4f}")

        # VALIDATION
        model.eval()
        val_loss = []
        
        # Reset metrics
        metric_precision.reset()
        metric_recall.reset()
        metric_f1.reset()
        metric_iou.reset()

        with torch.no_grad():
            val_bar = tqdm(val_dataloader, colour="blue", desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, masks in val_bar:
                images, masks = images.to(device), masks.to(device)
                
                outputs = model(images)
                preds = outputs['out']  # [Batch, Classes, Height, Width]
                
                v_loss = criterion(preds, masks)
                val_loss.append(v_loss.item())

                # Chuyển logits thành nhãn dự đoán cụ thể bằng argmax
                pred_labels = torch.argmax(preds, dim=1)

               
                metric_precision.update(pred_labels, masks)
                metric_recall.update(pred_labels, masks)
                metric_f1.update(pred_labels, masks)
                metric_iou.update(pred_labels, masks)

        precision_list = metric_precision.compute().cpu().tolist()
        recall_list = metric_recall.compute().cpu().tolist()
        f1_list = metric_f1.compute().cpu().tolist()
        iou_list = metric_iou.compute().cpu().tolist()
        
        # Mean IoU 
        miou = np.mean(iou_list)
        
        
        print("\n" + "="*60)
        print(f"--- THỐNG KÊ KẾT QUẢ VALIDATION (EPOCH {epoch+1}) ---")
        print(f"Val Loss: {np.mean(val_loss):.4f}")
        print(f"Mean IoU (mIoU): {miou:.4f}")
   
        for c in range(num_classes):
            print(f"  > Lớp {c}: Precision = {precision_list[c]:.4f} | Recall = {recall_list[c]:.4f} | F1-Score = {f1_list[c]:.4f} | IoU = {iou_list[c]:.4f}")
        print("="*60 + "\n")

        # Save model
        if miou > best_miou:
            best_miou = miou
            best_epoch = epoch 
            best_precision = precision_list.copy()
            best_recall = recall_list.copy()
            best_f1 = f1_list.copy()
            best_iou = iou_list.copy()
            if not os.path.exists("./model"):
                os.makedirs("./model", exist_ok=True)
            name = f"best_deeplabv3" + f"_{backbone}" + f"_segmentation.pth"
            path = os.path.join("./model", name)
            torch.save(model.state_dict(), path)
            print(f"Training completed. Best mIoU: {best_miou:.4f} at epoch {best_epoch + 1}. Model saved to './model/best_deeplabv3_{backbone}_segmentation.pth'.")
            
    return {
        "best_miou": best_miou,
        "best_epoch": best_epoch + 1,
        "precision per class in best epoch": best_precision,
        "recall per class in best epoch": best_recall,
        "f1 per class in best epoch": best_f1,
        "iou per class in best epoch": best_iou
    }
