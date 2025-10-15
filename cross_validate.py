import os
import torch
import torch.nn as nn
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from torch.utils.data import DataLoader, Subset
from dataset.multichannel_dataset import RamanMultiChannelDataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def plot_and_save_confusion_matrix(cm, labels, normalize=False, title='', save_path='cm.png'):
    cm_display = cm.astype('float')
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100
        fmt = ".2f"
    else:
        fmt = ".0f"

    annotations = np.empty_like(cm_display, dtype=object)
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            annotations[i, j] = f"{cm_display[i, j]:.2f}%" if normalize else f"{int(cm_display[i, j])}"

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_display, annot=annotations, fmt='', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def get_model(model_name):
    if model_name == 'resnet18':
        from models.resnet import ResNET18 as create_model
    elif model_name == 'efficientnet':
        from models.efficientnet5c import EfficientNet5C as create_model
    elif model_name == 'mobilevit':
        from models.mobilevit import mobile_vit_xx_small as create_model
    elif model_name == 'resnetse':
        from models.resnetse import ResNet18_SE as create_model
    elif model_name == 'remnet':
        from models.remnet import REMClassifier as create_model
    elif model_name == 'pinn':
        from models.pinn import RamanPINNClassifier as create_model
    else:
        raise ValueError(f"未知模型名称：{model_name}")
    return create_model(in_channels=5, num_classes=2)


def evaluate(model, dataloader, criterion, device, model_name):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels, signals in dataloader:
            if model_name == "pinn":
                images = signals
            if model_name == 'remnet':
                images, labels, signals = images.to(device), labels.to(device), signals.to(device)
                outputs = model(images, signals)
            else:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

            loss = criterion(outputs, labels)
            # if model_name == "pinn":
            #     loss_phy = model.physics_loss(images)
            #     loss = loss_cls + 0.5 * loss_phy
            # else:
            #     loss = loss_cls

            preds = torch.argmax(outputs, dim=1)

            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = total_correct / total_samples
    avg_loss = total_loss / total_samples
    return avg_loss, acc, y_true, y_pred


def cross_validate(
    image_root,
    method,
    model_name,
    epochs=30,
    batch_size=16,
    lr=1e-4,
    num_folds=5,
    save_dir="./weights_cv"
):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = RamanMultiChannelDataset(image_root,  method=method)
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    os.makedirs(save_dir, exist_ok=True)

    metrics_all = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n===== Fold {fold + 1}/{num_folds} =====")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = get_model(model_name).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # === Train ===
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for images, labels, signals in train_loader:
                if model_name == "pinn":
                    images = signals
                optimizer.zero_grad()
                if model_name == 'remnet':
                    images, labels, signals = images.to(device), labels.to(device), signals.to(device)
                    outputs = model(images, signals)
                else:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * images.size(0)

            avg_loss = total_loss / len(train_subset)
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_loss:.4f}")

        # === Validate ===
        val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion, device, model_name)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"[Fold {fold+1}] Accuracy: {val_acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        print(classification_report(y_true, y_pred, target_names=['nonPCOS', 'PCOS']))

        cm = confusion_matrix(y_true, y_pred)
        labels = ['nonPCOS', 'PCOS']
        cm_path = os.path.join(save_dir, f"{model_name}_fold{fold+1}_cm.png")
        plot_and_save_confusion_matrix(cm, labels, normalize=True,
                                       title=f"Fold {fold+1} Confusion Matrix",
                                       save_path=cm_path)

        metrics_all.append([val_acc, prec, rec, f1])
        torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_fold{fold+1}.pth"))

    # === 平均指标 ===
    metrics_all = np.array(metrics_all)
    mean_metrics = metrics_all.mean(axis=0)
    print("\n===== Cross Validation Summary =====")
    print(f"Average Accuracy:  {mean_metrics[0]:.4f}")
    print(f"Average Precision: {mean_metrics[1]:.4f}")
    print(f"Average Recall:    {mean_metrics[2]:.4f}")
    print(f"Average F1-score:  {mean_metrics[3]:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="5-Fold Cross Validation for PCOS Raman Classifier")
    parser.add_argument('--image_root', type=str, required=True, help='图像根目录')
    parser.add_argument('--method', type=str, default='rp', help='图像转换方法（rp, mtf, stft）')
    parser.add_argument('--model_name', type=str, default='remnet', help='模型名称')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='./weights_cv')
    args = parser.parse_args()

    cross_validate(
        image_root=args.image_root,
        method=args.method,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir
    )
