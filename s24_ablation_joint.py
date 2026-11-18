# ============================================================================
# Joint threshold + ROI expansion ablation (validation set).
# Repository name : s24_ablation_joint.py
# Original name   : 验证集_检测阈值与ROI联合消融_严格复现.py
# Paper           : Task-Decoupled Fine-Grained Product Recognition Using
#                   Tight Predicted ROIs for Automated Vending Cabinets
# Note            : algorithm unchanged; comments kept in the original Chinese.
# ============================================================================
from __future__ import annotations

import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import timm
import torch
from PIL import Image
from torchvision import datasets, transforms
from ultralytics import YOLO


# ============================================================
# 放在项目根目录运行。
#(repository root)
# 目的：
# 仅在验证集上联合选择检测阈值和ROI扩展比例，避免使用测试集调参。
#(repository root)
# 严格复现正式裁剪：
# 1. YOLO默认预测设置；
# 2. 选择最高置信度检测框；
# 3. xyxy先用int()截断；
# 4. 按比例扩展整数框；
# 5. 保存为JPG；
# 6. 分类器重新读取JPG分类。
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
    / "validation_joint_ablation_exact_crop"
)
CROPS_ROOT = RESULTS_DIR / "saved_jpg_rois"

THRESHOLDS = [0.10, 0.20, 0.25, 0.30]
EXPANSION_RATIOS = [0.00]

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


def ratio_name(ratio: float) -> str:
    return f"expand_{int(round(ratio * 100)):02d}pct"


def classify_saved_jpg(
    image_path: Path,
    classifier,
    transform,
    class_names: list[str],
    device: torch.device,
) -> tuple[str, float]:
    image = Image.open(image_path).convert("RGB")
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
    print("检测阈值：", THRESHOLDS)
    print(
        "ROI扩展比例：",
        [f"{ratio:.0%}" for ratio in EXPANSION_RATIOS],
    )

    for required_path, description in [
        (YOLO_MODEL_PATH, "YOLO权重"),
        (CLASSIFIER_MODEL_PATH, "分类器权重"),
        (TRAIN_DATA_DIR, "分类训练目录"),
        (VAL_IMAGE_DIR, "验证集图像目录"),
    ]:
        if not required_path.exists():
            raise FileNotFoundError(
                f"找不到{description}：{required_path}"
            )

    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_ROOT.mkdir(parents=True, exist_ok=True)

    crop_dirs = {}
    for ratio in EXPANSION_RATIOS:
        crop_dir = CROPS_ROOT / ratio_name(ratio)
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_dirs[ratio] = crop_dir

    temp_dataset = datasets.ImageFolder(str(TRAIN_DATA_DIR))
    class_names = list(temp_dataset.classes)
    sorted_classes = sorted(
        class_names,
        key=len,
        reverse=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
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
        load_state_dict_compat(
            CLASSIFIER_MODEL_PATH,
            device,
        ),
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
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    total_images = len(image_paths)

    print(f"验证图片数量：{total_images}")
    print("\n开始验证集联合消融……")

    # 每张图片只检测一次，每个扩展比例只分类一次。
    sample_records: list[dict[str, object]] = []

    for index, image_path in enumerate(image_paths, start=1):
        record: dict[str, object] = {
            "filename": image_path.name,
            "true_class": "",
            "detector_confidence": "",
            "detector_pred_class": "",
            "status": "",
        }

        true_class = infer_true_class(
            image_path,
            sorted_classes,
        )
        record["true_class"] = true_class or ""

        if true_class is None:
            record["status"] = "unknown_true_class"
            sample_records.append(record)
            continue

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            record["status"] = "image_read_failed"
            sample_records.append(record)
            continue

        height, width = image_bgr.shape[:2]

        try:
            result = detector(
                str(image_path),
                verbose=False,
            )[0]
            boxes = result.boxes
        except Exception as exc:
            record["status"] = "detector_failed"
            record["error"] = str(exc)
            sample_records.append(record)
            continue

        if boxes is None or len(boxes) == 0:
            record["status"] = "no_detection"
            sample_records.append(record)
            continue

        best_box = max(
            boxes,
            key=lambda box: box.conf[0],
        )
        det_confidence = float(best_box.conf[0])
        det_class_id = int(best_box.cls[0])
        det_pred_class = str(
            detector.names[det_class_id]
        )

        record["detector_confidence"] = det_confidence
        record["detector_pred_class"] = det_pred_class

        original_x1, original_y1, original_x2, original_y2 = map(
            int,
            best_box.xyxy[0].tolist(),
        )
        box_width = original_x2 - original_x1
        box_height = original_y2 - original_y1

        for ratio in EXPANSION_RATIOS:
            prefix = ratio_name(ratio)

            x1 = max(
                0,
                int(original_x1 - box_width * ratio),
            )
            y1 = max(
                0,
                int(original_y1 - box_height * ratio),
            )
            x2 = min(
                width,
                int(original_x2 + box_width * ratio),
            )
            y2 = min(
                height,
                int(original_y2 + box_height * ratio),
            )

            roi = image_bgr[y1:y2, x1:x2]
            if roi.size == 0:
                record[f"{prefix}_status"] = "empty_roi"
                continue

            save_name = f"{image_path.stem}_{prefix}.jpg"
            save_path = crop_dirs[ratio] / save_name

            if not cv2.imwrite(str(save_path), roi):
                record[f"{prefix}_status"] = "save_failed"
                continue

            try:
                pred_class, cls_confidence = classify_saved_jpg(
                    save_path,
                    classifier,
                    transform,
                    class_names,
                    device,
                )
            except Exception as exc:
                record[f"{prefix}_status"] = "classifier_failed"
                record[f"{prefix}_error"] = str(exc)
                continue

            record[f"{prefix}_pred_class"] = pred_class
            record[f"{prefix}_classifier_confidence"] = (
                cls_confidence
            )
            record[f"{prefix}_correct"] = (
                pred_class == true_class
            )
            record[f"{prefix}_status"] = "ok"

        record["status"] = "processed"
        sample_records.append(record)

        if index % 100 == 0:
            print(f"已处理：{index}/{total_images}")

    # 保存每张图片的基础结果
    all_fieldnames = sorted(
        {
            key
            for record in sample_records
            for key in record.keys()
        }
    )
    sample_csv_path = RESULTS_DIR / "sample_predictions.csv"
    with sample_csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=all_fieldnames,
        )
        writer.writeheader()
        writer.writerows(sample_records)

    grid_rows: list[dict[str, object]] = []

    for threshold in THRESHOLDS:
        for ratio in EXPANSION_RATIOS:
            prefix = ratio_name(ratio)

            covered = 0
            correct = 0
            unknown_true_class = 0
            failed = 0

            for record in sample_records:
                true_class = record.get("true_class", "")
                if not true_class:
                    unknown_true_class += 1
                    continue

                det_conf = record.get(
                    "detector_confidence",
                    "",
                )
                if det_conf == "":
                    continue

                if float(det_conf) < threshold:
                    continue

                if record.get(f"{prefix}_status") != "ok":
                    failed += 1
                    continue

                covered += 1
                correct += int(
                    bool(record.get(f"{prefix}_correct"))
                )

            missed = total_images - covered
            coverage = (
                covered / total_images
                if total_images
                else 0.0
            )
            conditional_accuracy = (
                correct / covered
                if covered
                else 0.0
            )
            end_to_end_accuracy = (
                correct / total_images
                if total_images
                else 0.0
            )

            grid_rows.append(
                {
                    "confidence_threshold": threshold,
                    "roi_expansion_ratio": ratio,
                    "roi_expansion_percent": int(
                        round(ratio * 100)
                    ),
                    "total_images": total_images,
                    "covered_images": covered,
                    "missed_or_below_threshold": missed,
                    "correct_images": correct,
                    "detection_coverage": coverage,
                    "conditional_classification_accuracy": (
                        conditional_accuracy
                    ),
                    "complete_end_to_end_accuracy": (
                        end_to_end_accuracy
                    ),
                    "unknown_true_class": unknown_true_class,
                    "failed_after_detection": failed,
                }
            )

    # 先按端到端准确率排序；并列时优先条件准确率、覆盖率，
    # 再优先更紧的ROI和更低的检测阈值。
    ranked_rows = sorted(
        grid_rows,
        key=lambda row: (
            -float(row["complete_end_to_end_accuracy"]),
            -float(
                row["conditional_classification_accuracy"]
            ),
            -float(row["detection_coverage"]),
            float(row["roi_expansion_ratio"]),
            float(row["confidence_threshold"]),
        ),
    )

    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank

    grid_csv_path = RESULTS_DIR / "validation_grid_summary.csv"
    fieldnames = [
        "rank",
        "confidence_threshold",
        "roi_expansion_ratio",
        "roi_expansion_percent",
        "total_images",
        "covered_images",
        "missed_or_below_threshold",
        "correct_images",
        "detection_coverage",
        "conditional_classification_accuracy",
        "complete_end_to_end_accuracy",
        "unknown_true_class",
        "failed_after_detection",
    ]
    with grid_csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        for row in ranked_rows:
            writer.writerow(
                {
                    field: row.get(field, "")
                    for field in fieldnames
                }
            )

    best = ranked_rows[0]
    best_config = {
        "selection_split": "validation",
        "selection_metric": (
            "complete_end_to_end_accuracy"
        ),
        "confidence_threshold": best[
            "confidence_threshold"
        ],
        "roi_expansion_ratio": best[
            "roi_expansion_ratio"
        ],
        "roi_expansion_percent": best[
            "roi_expansion_percent"
        ],
        "covered_images": best["covered_images"],
        "correct_images": best["correct_images"],
        "detection_coverage": best[
            "detection_coverage"
        ],
        "conditional_classification_accuracy": best[
            "conditional_classification_accuracy"
        ],
        "complete_end_to_end_accuracy": best[
            "complete_end_to_end_accuracy"
        ],
    }

    best_config_path = RESULTS_DIR / "best_config.json"
    best_config_path.write_text(
        json.dumps(
            best_config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "experiment": (
            "validation_joint_threshold_roi_ablation"
        ),
        "selection_split": "validation",
        "total_images": total_images,
        "thresholds": THRESHOLDS,
        "expansion_ratios": EXPANSION_RATIOS,
        "best_config": best_config,
        "all_results": ranked_rows,
    }

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n✅ 验证集联合消融完成")
    print("排名前5的配置：")
    print(
        "排名 | 阈值 | 扩展 | 覆盖数 | 正确数 | "
        "条件准确率 | 端到端准确率"
    )

    for row in ranked_rows[:5]:
        print(
            f"{row['rank']:>2d} | "
            f"{row['confidence_threshold']:.2f} | "
            f"{row['roi_expansion_percent']:>2d}% | "
            f"{row['covered_images']} | "
            f"{row['correct_images']} | "
            f"{row['conditional_classification_accuracy'] * 100:.2f}% | "
            f"{row['complete_end_to_end_accuracy'] * 100:.2f}%"
        )

    print("\n验证集选出的最佳配置：")
    print(
        f"检测阈值："
        f"{best_config['confidence_threshold']:.2f}"
    )
    print(
        f"ROI扩展："
        f"{best_config['roi_expansion_percent']}%"
    )
    print(
        f"验证集端到端准确率："
        f"{best_config['complete_end_to_end_accuracy'] * 100:.2f}%"
    )
    print("\n结果目录：", RESULTS_DIR)
    print("联合结果：", grid_csv_path)
    print("最佳配置：", best_config_path)
    print("完整汇总：", summary_path)


if __name__ == "__main__":
    main()
