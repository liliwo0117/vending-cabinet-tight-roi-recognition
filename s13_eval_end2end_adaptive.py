# ============================================================================
# End-to-end evaluation with adaptive ROI expansion.
# Repository name : s13_eval_end2end_adaptive.py
# Original name   : eval_end2end_adaptive.py
# Paper           : Task-Decoupled Fine-Grained Product Recognition Using
#                   Tight Predicted ROIs for Automated Vending Cabinets
# Note            : algorithm unchanged; comments kept in the original Chinese.
# ============================================================================
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import timm
import torch
from PIL import Image
from torchvision import datasets, transforms


# 本文件请放在项目最外层：
# shiyan2-commodity_detect\eval_end2end_我的GPU版.py
PROJECT_ROOT = Path(__file__).resolve().parent

PRED_ROI_DIR = PROJECT_ROOT / "cropped_pred_adaptive_我的GPU版"
MODEL_PATH = PROJECT_ROOT / "best_classifier_我的GPU版.pth"
TRAIN_DATA_DIR = PROJECT_ROOT / "cls_data_new" / "train"
RESULTS_DIR = PROJECT_ROOT / "results" / "end2end_adaptive_我的GPU版"
ERROR_DIR = RESULTS_DIR / "error_samples"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_state_dict_compat(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    if not PRED_ROI_DIR.is_dir():
        raise FileNotFoundError(f"找不到预测 ROI 目录：{PRED_ROI_DIR}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"找不到你的分类模型：{MODEL_PATH}")
    if not TRAIN_DATA_DIR.is_dir():
        raise FileNotFoundError(f"找不到分类训练目录：{TRAIN_DATA_DIR}")

    temp_dataset = datasets.ImageFolder(str(TRAIN_DATA_DIR))
    class_names = list(temp_dataset.classes)
    sorted_classes = sorted(class_names, key=len, reverse=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备：", device)
    if device.type == "cuda":
        print("显卡：", torch.cuda.get_device_name(0))

    model = timm.create_model(
        "convnextv2_atto",
        pretrained=False,
        num_classes=len(class_names),
    )
    model.load_state_dict(load_state_dict_compat(MODEL_PATH, device), strict=True)
    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    correct = 0
    skipped_no_label = 0
    failed = 0
    rows = []

    print("类别顺序：", class_names)
    print("开始端到端分类评估……")

    for image_path in sorted(PRED_ROI_DIR.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        name_clean = image_path.stem.split("(")[0]
        true_class = next(
            (name for name in sorted_classes if name_clean.startswith(name)),
            None,
        )
        if true_class is None:
            skipped_no_label += 1
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor)
                probabilities = torch.softmax(logits, dim=1)
                pred_index = int(probabilities.argmax(dim=1).item())
                confidence = float(probabilities[0, pred_index].item())

            pred_class = class_names[pred_index]
            is_correct = pred_class == true_class

            total += 1
            correct += int(is_correct)
            rows.append(
                {
                    "filename": image_path.name,
                    "true_class": true_class,
                    "pred_class": pred_class,
                    "confidence": confidence,
                    "correct": is_correct,
                }
            )

            if not is_correct:
                shutil.copy2(image_path, ERROR_DIR / image_path.name)

        except Exception as exc:
            failed += 1
            rows.append(
                {
                    "filename": image_path.name,
                    "true_class": true_class,
                    "pred_class": "",
                    "confidence": "",
                    "correct": False,
                    "error": str(exc),
                }
            )

    accuracy = correct / total if total else 0.0

    csv_path = RESULTS_DIR / "predictions.csv"
    fieldnames = [
        "filename",
        "true_class",
        "pred_class",
        "confidence",
        "correct",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    summary = {
        "total_evaluated": total,
        "correct": correct,
        "accuracy": accuracy,
        "skipped_no_label": skipped_no_label,
        "failed": failed,
        "class_names": class_names,
        "model_path": str(MODEL_PATH),
        "pred_roi_dir": str(PRED_ROI_DIR),
    }
    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 50)
    print("✅ 端到端评估完成")
    print(f"总评估图片数：{total}")
    print(f"分类正确数：{correct}")
    print(f"端到端准确率：{accuracy * 100:.2f}%")
    print(f"无法解析真实类别而跳过：{skipped_no_label}")
    print(f"读取或推理失败：{failed}")
    print("结果目录：", RESULTS_DIR)
    print("=" * 50)


if __name__ == "__main__":
    main()
