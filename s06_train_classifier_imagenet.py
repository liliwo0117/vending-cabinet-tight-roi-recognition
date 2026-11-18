# ============================================================================
# Train ConvNeXt V2-Atto from official ImageNet weights (paper).
# Repository name : s06_train_classifier_imagenet.py
# Original name   : train_classifier_ImageNet正式版.py
# Paper           : Task-Decoupled Fine-Grained Product Recognition Using
#                   Tight Predicted ROIs for Automated Vending Cabinets
# Note            : algorithm unchanged; comments kept in the original Chinese.
# ============================================================================
from __future__ import annotations

import os
import json
from pathlib import Path

# ✅ 一定要放在 import timm 前面
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

from tqdm import tqdm
import matplotlib.pyplot as plt
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 本文件请放在项目最外层：
# shiyan2-commodity_detect\train_classifier_我的GPU版.py
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "cls_data_new"
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "classifier_imagenet_official"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = RESULTS_DIR / "best_model_imagenet.pth"
CLASS_NAMES_PATH = RESULTS_DIR / "class_names_imagenet.json"

EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
PATIENCE = 5





def main() -> None:
    if not TRAIN_DIR.is_dir():
        raise FileNotFoundError(f"找不到分类训练目录：{TRAIN_DIR}")
    if not VAL_DIR.is_dir():
        raise FileNotFoundError(f"找不到分类验证目录：{VAL_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备：", device)
    if device.type == "cuda":
        print("显卡：", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
    else:
        raise RuntimeError("当前 Python 环境没有检测到 CUDA，请检查 PyCharm 解释器。")

    transform_train = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=transform_train)
    val_dataset = datasets.ImageFolder(str(VAL_DIR), transform=transform_val)

    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            f"train/val 类别文件夹不一致：train={train_dataset.classes}, "
            f"val={val_dataset.classes}"
        )

    class_names = list(train_dataset.classes)
    CLASS_NAMES_PATH.write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"类别数：{len(class_names)}")
    print(f"类别顺序：{class_names}")
    print(f"训练集：{len(train_dataset)} 张")
    print(f"验证集：{len(val_dataset)} 张")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = timm.create_model(
        model_name="hf_hub:timm/convnextv2_atto.fcmae_ft_in1k",
        pretrained=True,
        num_classes=len(class_names),
    )

    print("模型初始化方式：官方 ImageNet-1K 预训练权重")
    print("模型名称：convnextv2_atto.fcmae_ft_in1k")

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    train_losses = []
    val_accs = []
    best_acc = -1.0
    no_improve = 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        total_train = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_count = labels.size(0)
            running_loss += float(loss.item()) * batch_count
            total_train += batch_count
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(1, total_train)
        train_losses.append(train_loss)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                predictions = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += int((predictions == labels).sum().item())

        val_acc = correct / max(1, total)
        val_accs.append(val_acc)
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, "
            f"val_acc={val_acc * 100:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("保存新的最佳模型：", BEST_MODEL_PATH)
        else:
            no_improve += 1
            print(f"验证准确率未提升：{no_improve}/{PATIENCE}")
            if no_improve >= PATIENCE:
                print("触发早停。")
                break

    metrics = {
        "best_val_acc": best_acc,
        "train_losses": train_losses,
        "val_accs": val_accs,
        "class_names": class_names,
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    epochs_ran = list(range(1, len(train_losses) + 1))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs_ran, train_losses, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2 = ax1.twinx()
    ax2.plot(epochs_ran, [x * 100 for x in val_accs], label="Val Accuracy")
    ax2.set_ylabel("Accuracy (%)")

    plt.title("ConvNeXt V2 Training Curves")
    fig.tight_layout()
    curve_path = RESULTS_DIR / "training_curves.png"
    plt.savefig(curve_path, dpi=300)
    plt.close(fig)

    print("\n✅ 分类模型训练完成")
    print(f"最佳验证准确率：{best_acc * 100:.2f}%")
    print("最佳模型：", BEST_MODEL_PATH)
    print("类别顺序：", CLASS_NAMES_PATH)
    print("训练结果：", RESULTS_DIR)


if __name__ == "__main__":
    main()
