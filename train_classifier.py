import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.multichannel_dataset import RamanMultiChannelDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from datetime import datetime
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import numpy as np
import torch.backends.cudnn as cudnn

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 保证卷积等操作的确定性
    cudnn.deterministic = True
    cudnn.benchmark = False


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
        from models.pinn import RamanTransformerClassifier as create_model
    else:
        print("输入模型名字错误，请检查重试")
    model = create_model(in_channels=5, num_classes=2)
    return model

def evaluate(model, dataloader, criterion, device,model_name):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels,signals in dataloader:
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

def train(
    image_root,
    method='rp',
    epochs=20,
    batch_size=16,
    lr=1e-4,
    log_dir='runs/classifier',
    save_path='classifier.pth',
    seed=42,
    model_name='resnet18',
    lambda_phy=0.1,
    use_pinn=False
):
    set_seed(seed)  # 设置固定随机种子
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据加载
    train_ds = RamanMultiChannelDataset(image_root, phase='train', method=method)
    test_ds = RamanMultiChannelDataset(image_root, phase='test', method=method)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 模型准备
    model = get_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # TensorBoard记录器
    log_path = os.path.join(log_dir, datetime.now().strftime('%Y%m%d-%H%M%S'))
    writer = SummaryWriter(log_path)

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss, total_train_correct, total_samples = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}]")
        for images, labels, signals in pbar:
            if model_name == "pinn":
                images = signals
            if model_name == 'remnet':
                images, labels, signals = images.to(device), labels.to(device), signals.to(device)
                optimizer.zero_grad()
                outputs = model(images,signals)

            else:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = model(images)
            loss_cls  = criterion(outputs, labels)

            if use_pinn and hasattr(model, 'physics_loss'):
                loss_phy = model.physics_loss(images)
                loss = loss_cls + lambda_phy * loss_phy
            else:
                loss = loss_cls

            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            total_train_loss += loss.item() * images.size(0)
            total_train_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = total_train_loss / total_samples
        train_acc = total_train_correct / total_samples
        test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device,model_name)

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('Acc/Train', train_acc, epoch)
        writer.add_scalar('Acc/Test', test_acc, epoch)

        print(f"[✓] Epoch {epoch}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")


    print("\n[Classification Report]")
    print(classification_report(y_true, y_pred, target_names=['nonPCOS', 'PCOS']))
    print("[Confusion Matrix]")
    print(confusion_matrix(y_true, y_pred))

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"[✓] Final Test Accuracy: {acc:.4f}")
    print(f"[✓] Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"✅ 模型已保存: {save_path}")
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train PCOS Raman classifier")
    parser.add_argument('--image_root', type=str, required=True, help='图像根目录')
    parser.add_argument('--method', type=str, default='rp', help='图像转换方法，例如 rp, mtf, stft')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='每个batch大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--log_dir', type=str, default='runs/classifier', help='TensorBoard记录路径')
    parser.add_argument('--save_path', type=str, default='classifier.pth', help='模型保存路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--model_name', type=str, default='resnet', help='默认模型: 0.resnet18\n\r1.efficientnet\n\r2.mobilevit\n\r3.mobilevit\n\r4.resnetse\n\r5.remnet\n\r6.pinn')
    parser.add_argument('--use_pinn', action='store_true', help='是否启用物理信息神经网络（PINN）损失')
    parser.add_argument('--lambda_phy', type=float, default=0.1, help='物理损失权重 λ')

    args = parser.parse_args()

    train(
        image_root=args.image_root,
        method=args.method,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        log_dir=args.log_dir,
        save_path=args.save_path,
        seed=args.seed,
        lambda_phy=args.lambda_phy,
        model_name=args.model_name
    )

