# ============================================================================
# Boundary check around very low detection thresholds.
# Repository name : s25_ablation_low_threshold_boundary.py
# Original name   : 验证集低阈值边界检查_修正版.py
# Paper           : Task-Decoupled Fine-Grained Product Recognition Using
#                   Tight Predicted ROIs for Automated Vending Cabinets
# Note            : algorithm unchanged; comments kept in the original Chinese.
# ============================================================================
from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import cv2
import timm
import torch
from PIL import Image
from torchvision import datasets, transforms
from ultralytics import YOLO


# ============================================================
# 将本脚本放在项目根目录运行。
#(repository root)
# 目的：
# 在验证集上正确测试低于0.25的检测置信度阈值。
#(repository root)
# 重要：
# Ultralytics默认预测conf约为0.25。若不显式降低该参数，
# 0.10、0.20和0.25会得到相同结果，因为低于0.25的框
# 已在YOLO内部被过滤。
#(repository root)
# 本脚本先以0.01获取候选框，再分别评价：
# 0.01、0.05、0.10、0.15、0.20、0.25、0.30。
#(repository root)
# ROI扩展固定为0%，裁剪方式与正式流程一致：
# xyxy先int()截断 -> 保存JPG -> 重新读取分类。
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

YOLO_MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "train_8cls_gpu2"
    / "weights"
    / "best.pt"
)

CLASSIFIER_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "classifier_imagenet_official"
    / "best_model_imagenet.pth"
)

TRAIN_DATA_DIR = PROJECT_ROOT / "cls_data_new" / "train"
VAL_IMAGE_DIR = PROJECT_ROOT / "data" / "val_images"

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "validation_low_threshold_ablation_corrected"
)
ROI_DIR = RESULTS_DIR / "saved_jpg_rois_0pct"

THRESHOLDS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
DETECTOR_MIN_CONF = min(THRESHOLDS)
ROI_EXPANSION = 0.0

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_state_dict_compat(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def infer_true_class(
    image_path: Path,
    sorted_classes: list[str],
) -> str | None:
    name_clean = image_path.stem.split("(")[0]
    return next(
        (
            class_name
            for class_name in sorted_classes
            if name_clean.startswith(class_name)
        ),
        None,
    )


def classify_saved_jpg(
    saved_path: Path,
    classifier,
    transform,
    class_names: list[str],
    device: torch.device,
) -> tuple[str, float]:
    image = Image.open(saved_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = classifier(tensor)
        probabilities = torch.softmax(logits, dim=1)
        pred_index = int(probabilities.argmax(dim=1).item())
        confidence = float(probabilities[0, pred_index].item())

    return class_names[pred_index], confidence


def main() -> None:
    print("YOLO权重：", YOLO_MODEL_PATH)
    print("分类器权重：", CLASSIFIER_MODEL_PATH)
    print("验证集目录：", VAL_IMAGE_DIR)
    print("结果目录：", RESULTS_DIR)
    print("候选框最低置信度：", DETECTOR_MIN_CONF)
    print("评价阈值：", THRESHOLDS)
    print("ROI扩展比例：0%")

    for path, description in [
        (YOLO_MODEL_PATH, "YOLO权重"),
        (CLASSIFIER_MODEL_PATH, "分类器权重"),
        (TRAIN_DATA_DIR, "分类训练目录"),
        (VAL_IMAGE_DIR, "验证集目录"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"找不到{description}：{path}")

    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ROI_DIR.mkdir(parents=True, exist_ok=True)

    dataset = datasets.ImageFolder(str(TRAIN_DATA_DIR))
    class_names = list(dataset.classes)
    sorted_classes = sorted(class_names, key=len, reverse=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_device: int | str = 0 if torch.cuda.is_available() else "cpu"

    print("设备：", device)
    if device.type == "cuda":
        print("显卡：", torch.cuda.get_device_name(0))
    print("类别顺序：", class_names)

    detector = YOLO(str(YOLO_MODEL_PATH))

    classifier = timm.create_model(
        "convnextv2_atto.fcmae_ft_in1k",
        pretrained=False,
        num_classes=len(class_names),
    )
    classifier.load_state_dict(
        load_state_dict_compat(CLASSIFIER_MODEL_PATH, device),
        strict=True,
    )
    classifier.to(device)
    classifier.eval()

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

    image_paths = sorted(
        path
        for path in VAL_IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    total_images = len(image_paths)
    print("验证图片数量：", total_images)

    sample_rows: list[dict[str, object]] = []

    print("\n开始低阈值边界检查……")

    for index, image_path in enumerate(image_paths, start=1):
        true_class = infer_true_class(image_path, sorted_classes)

        row: dict[str, object] = {
            "filename": image_path.name,
            "true_class": true_class or "",
            "detector_confidence": "",
            "detector_pred_class": "",
            "classifier_pred_class": "",
            "classifier_confidence": "",
            "classifier_correct": False,
            "status": "",
            "error": "",
        }

        if true_class is None:
            row["status"] = "unknown_true_class"
            sample_rows.append(row)
            continue

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            row["status"] = "image_read_failed"
            sample_rows.append(row)
            continue

        height, width = image_bgr.shape[:2]

        try:
            result = detector.predict(
                source=str(image_path),
                conf=DETECTOR_MIN_CONF,
                device=yolo_device,
                save=False,
                verbose=False,
            )[0]
            boxes = result.boxes
        except Exception as exc:
            row["status"] = "detector_failed"
            row["error"] = str(exc)
            sample_rows.append(row)
            continue

        if boxes is None or len(boxes) == 0:
            row["status"] = "no_detection_at_min_conf"
            sample_rows.append(row)
            continue

        best_box = max(boxes, key=lambda box: box.conf[0])
        detector_confidence = float(best_box.conf[0])
        detector_class_id = int(best_box.cls[0])
        detector_pred_class = str(detector.names[detector_class_id])

        row["detector_confidence"] = detector_confidence
        row["detector_pred_class"] = detector_pred_class

        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())

        # ROI扩展固定0%，保留正式流程中的整数坐标截断。
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        roi = image_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            row["status"] = "empty_roi"
            sample_rows.append(row)
            continue

        save_path = ROI_DIR / f"{image_path.stem}_tight.jpg"
        if not cv2.imwrite(str(save_path), roi):
            row["status"] = "save_failed"
            sample_rows.append(row)
            continue

        try:
            pred_class, classifier_confidence = classify_saved_jpg(
                save_path,
                classifier,
                transform,
                class_names,
                device,
            )
        except Exception as exc:
            row["status"] = "classifier_failed"
            row["error"] = str(exc)
            sample_rows.append(row)
            continue

        row["classifier_pred_class"] = pred_class
        row["classifier_confidence"] = classifier_confidence
        row["classifier_correct"] = pred_class == true_class
        row["status"] = "processed"
        sample_rows.append(row)

        if index % 100 == 0:
            print(f"已处理：{index}/{total_images}")

    sample_csv_path = RESULTS_DIR / "sample_predictions.csv"
    fieldnames = [
        "filename",
        "true_class",
        "detector_confidence",
        "detector_pred_class",
        "classifier_pred_class",
        "classifier_confidence",
        "classifier_correct",
        "status",
        "error",
    ]
    with sample_csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_rows)

    summary_rows: list[dict[str, object]] = []

    for threshold in THRESHOLDS:
        covered = 0
        correct = 0
        failed = 0
        unknown = 0

        for row in sample_rows:
            if not row["true_class"]:
                unknown += 1
                continue

            if row["status"] != "processed":
                if row["status"] not in {"no_detection_at_min_conf"}:
                    failed += 1
                continue

            detector_confidence = float(row["detector_confidence"])
            if detector_confidence < threshold:
                continue

            covered += 1
            correct += int(bool(row["classifier_correct"]))

        missed = total_images - covered
        coverage = covered / total_images if total_images else 0.0
        conditional_accuracy = correct / covered if covered else 0.0
        end_to_end_accuracy = correct / total_images if total_images else 0.0

        summary_rows.append(
            {
                "confidence_threshold": threshold,
                "detector_min_conf": DETECTOR_MIN_CONF,
                "roi_expansion_ratio": ROI_EXPANSION,
                "total_images": total_images,
                "covered_images": covered,
                "missed_or_below_threshold": missed,
                "correct_images": correct,
                "detection_coverage": coverage,
                "conditional_classification_accuracy": conditional_accuracy,
                "complete_end_to_end_accuracy": end_to_end_accuracy,
                "unknown_true_class": unknown,
                "failed": failed,
            }
        )

    ranked_rows = sorted(
        summary_rows,
        key=lambda row: (
            -float(row["complete_end_to_end_accuracy"]),
            -float(row["conditional_classification_accuracy"]),
            -float(row["detection_coverage"]),
            float(row["confidence_threshold"]),
        ),
    )

    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank

    summary_csv_path = RESULTS_DIR / "low_threshold_summary.csv"
    summary_fieldnames = [
        "rank",
        "confidence_threshold",
        "detector_min_conf",
        "roi_expansion_ratio",
        "total_images",
        "covered_images",
        "missed_or_below_threshold",
        "correct_images",
        "detection_coverage",
        "conditional_classification_accuracy",
        "complete_end_to_end_accuracy",
        "unknown_true_class",
        "failed",
    ]
    with summary_csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=summary_fieldnames)
        writer.writeheader()
        for row in ranked_rows:
            writer.writerow(
                {
                    field: row.get(field, "")
                    for field in summary_fieldnames
                }
            )

    best = ranked_rows[0]
    best_config = {
        "selection_split": "validation",
        "selection_metric": "complete_end_to_end_accuracy",
        "detector_min_conf": DETECTOR_MIN_CONF,
        "confidence_threshold": best["confidence_threshold"],
        "roi_expansion_ratio": ROI_EXPANSION,
        "roi_expansion_percent": 0,
        "covered_images": best["covered_images"],
        "correct_images": best["correct_images"],
        "detection_coverage": best["detection_coverage"],
        "conditional_classification_accuracy": best[
            "conditional_classification_accuracy"
        ],
        "complete_end_to_end_accuracy": best[
            "complete_end_to_end_accuracy"
        ],
    }

    best_config_path = RESULTS_DIR / "best_config.json"
    best_config_path.write_text(
        json.dumps(best_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "experiment": "validation_low_threshold_ablation_corrected",
                "detector_min_conf": DETECTOR_MIN_CONF,
                "thresholds": THRESHOLDS,
                "roi_expansion_ratio": ROI_EXPANSION,
                "total_images": total_images,
                "best_config": best_config,
                "all_results": ranked_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n✅ 修正版低阈值边界检查完成")
    print(
        "排名 | 阈值 | 覆盖数 | 正确数 | "
        "条件准确率 | 端到端准确率"
    )
    for row in ranked_rows:
        print(
            f"{row['rank']:>2d} | "
            f"{row['confidence_threshold']:.2f} | "
            f"{row['covered_images']} | "
            f"{row['correct_images']} | "
            f"{row['conditional_classification_accuracy'] * 100:.2f}% | "
            f"{row['complete_end_to_end_accuracy'] * 100:.2f}%"
        )

    print("\n验证集选出的最佳配置：")
    print(f"检测阈值：{best_config['confidence_threshold']:.2f}")
    print("ROI扩展：0%")
    print(
        f"验证集端到端准确率："
        f"{best_config['complete_end_to_end_accuracy'] * 100:.2f}%"
    )
    print("\n结果目录：", RESULTS_DIR)
    print("结果汇总：", summary_csv_path)
    print("最佳配置：", best_config_path)
    print("完整汇总：", summary_path)


if __name__ == "__main__":
    main()
