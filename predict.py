# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.multichannel_dataset import RamanMultiChannelDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import argparse
import numpy as np
import random
import torch.backends.cudnn as cudnn

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def plot_and_save_confusion_matrix(cm, labels, normalize=False, title='Confusion Matrix', save_path='confusion_matrix.png'):
    """
    绘制并保存混淆矩阵图像
    """
    cm_display = cm.astype('float')
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100
        fmt = ".2f"
    else:
        fmt = ".0f"

    # 手动构造带百分号的注释文本
    annotations = np.empty_like(cm_display, dtype=object)
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            annotations[i, j] = f"{cm_display[i, j]:.2f}%"

    # 绘图
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
                outputs = model(images,signals)
            else:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)
            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = total_correct / total_samples
    avg_loss = total_loss / total_samples
    return avg_loss, acc, y_true, y_pred



def predict(
    image_root,
    method,
    weight_path,
    batch_size,
    model_name
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_ds = RamanMultiChannelDataset(image_root, phase='test', method=method)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = get_model(model_name)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device, model_name)

    print("\n[Classification Report]")
    print(classification_report(y_true, y_pred, target_names=['nonPCOS', 'PCOS']))
    print("[Confusion Matrix]")
    print(confusion_matrix(y_true, y_pred))

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"[✓] Test Accuracy: {acc:.4f}")
    print(f"[✓] Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")
    # 保存混淆矩阵图
    cm = confusion_matrix(y_true, y_pred)
    labels = ['nonPCOS', 'PCOS']
    base_path = os.path.dirname(weight_path)
    save_name_base = f"{model_name}_{method}"
    dataset_name = image_root.split("_")[-1]
    plot_and_save_confusion_matrix(cm, labels, normalize=True,
                                    title=f"{model_name.upper()} - {dataset_name} Confusion Matrix",
                                    save_path=os.path.join(base_path, f"{save_name_base}_{dataset_name}_cm_percent.png"))
    print(f"[✓] 混淆矩阵图已保存至：{base_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict using trained PCOS classifier")
    parser.add_argument('--image_root', type=str, required=True, help='图像根目录')
    parser.add_argument('--method', type=str, default='rp', help='图像转换方法（rp, mtf, stft）')
    parser.add_argument('--batch_size', type=int, default=16, help='batch大小')
    parser.add_argument('--weight_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--model_name', type=str, default='remnet', help='模型名（resnet18, efficientnet, mobilevit, resnetse, remnet, pinn）')

    args = parser.parse_args()

    predict(
        image_root=args.image_root,
        method=args.method,
        weight_path=args.weight_path,
        batch_size=args.batch_size,
        model_name=args.model_name
    )
