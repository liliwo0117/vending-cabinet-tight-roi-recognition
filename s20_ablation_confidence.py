# ============================================================================
# Detection-confidence threshold ablation (paper Fig. 5).
# Repository name : s20_ablation_confidence.py
# Original name   : 检测置信度阈值消融_严格复现正式裁剪.py
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
# 将本脚本放在项目根目录运行：
# 
#(repository root)
# 本版本严格复现 crop_pred_final.py 的裁剪流程：
# 1. YOLO 使用 model(img_path, verbose=False)[0]
# 2. 从所有检测框中选置信度最高框
# 3. 先将 xyxy 浮点坐标用 int() 截断
# 4. 再按整数框向四周扩展 15%
# 5. 使用 cv2.imwrite 保存为 JPG
# 6. 分类器重新读取保存后的 JPG 进行分类
#(repository root)
# 唯一变化量：检测置信度阈值
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
TEST_IMAGE_DIR = PROJECT_ROOT / "data" / "test"

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "confidence_threshold_ablation_exact_crop"
)

CROPS_ROOT = RESULTS_DIR / "saved_jpg_rois"

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]
EXPAND_RATIO = 0.15
EXPECTED_TEST_IMAGE_TOTAL = 680

IMAGE_EXTENSIONS = {".jpg"}


def load_state_dict_compat(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def infer_true_class(
    image_path: Path,
    sorted_classes: list[str],
) -> str | None:
    # 与正式评价脚本保持一致：遇到 "(1)" 时取前面的原始名字。
    name_clean = image_path.stem.split("(")[0]
    return next(
        (
            class_name
            for class_name in sorted_classes
            if name_clean.startswith(class_name)
        ),
        None,
    )


def threshold_folder_name(threshold: float) -> str:
    return f"conf_{threshold:.2f}".replace(".", "_")


def prepare_output_dirs() -> dict[float, Path]:
    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_ROOT.mkdir(parents=True, exist_ok=True)

    crop_dirs: dict[float, Path] = {}
    for threshold in THRESHOLDS:
        crop_dir = CROPS_ROOT / threshold_folder_name(threshold)
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_dirs[threshold] = crop_dir

    return crop_dirs


def classify_saved_jpg(
    saved_roi_path: Path,
    classifier,
    transform,
    class_names: list[str],
    device: torch.device,
) -> tuple[str, float]:
    image = Image.open(saved_roi_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = classifier(tensor)
        probabilities = torch.softmax(logits, dim=1)
        pred_index = int(probabilities.argmax(dim=1).item())
        confidence = float(probabilities[0, pred_index].item())

    return class_names[pred_index], confidence


def write_predictions(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    fieldnames = [
        "filename",
        "saved_roi_filename",
        "true_class",
        "detector_pred_class",
        "detector_confidence",
        "classifier_pred_class",
        "classifier_confidence",
        "covered",
        "correct",
        "status",
        "error",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: row.get(field, "")
                    for field in fieldnames
                }
            )


def main() -> None:
    print("YOLO权重：", YOLO_MODEL_PATH)
    print("分类器权重：", CLASSIFIER_MODEL_PATH)
    print("测试集目录：", TEST_IMAGE_DIR)
    print("结果目录：", RESULTS_DIR)
    print("阈值：", THRESHOLDS)
    print(f"ROI扩展比例：{EXPAND_RATIO:.0%}")
    print("裁剪方式：严格复现 crop_pred_final.py")
    print("分类输入：保存为JPG后重新读取")

    if not YOLO_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"找不到YOLO权重：{YOLO_MODEL_PATH}"
        )
    if not CLASSIFIER_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"找不到分类器权重：{CLASSIFIER_MODEL_PATH}"
        )
    if not TRAIN_DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"找不到分类训练目录：{TRAIN_DATA_DIR}"
        )
    if not TEST_IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"找不到测试图片目录：{TEST_IMAGE_DIR}"
        )

    crop_dirs = prepare_output_dirs()

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

    # 原脚本仅处理 .jpg
    image_names = [
        name
        for name in os.listdir(TEST_IMAGE_DIR)
        if Path(name).suffix.lower() in IMAGE_EXTENSIONS
    ]

    total_images = len(image_names)
    if total_images != EXPECTED_TEST_IMAGE_TOTAL:
        print(
            f"⚠ 当前找到 {total_images} 张JPG，"
            f"预期为 {EXPECTED_TEST_IMAGE_TOTAL} 张。"
        )

    rows_by_threshold = {
        threshold: []
        for threshold in THRESHOLDS
    }

    stats_by_threshold = {
        threshold: {
            "covered": 0,
            "correct": 0,
            "unknown_true_class": 0,
            "image_read_failed": 0,
            "detector_failed": 0,
            "empty_roi": 0,
            "save_failed": 0,
            "classifier_failed": 0,
        }
        for threshold in THRESHOLDS
    }

    class_stats = {
        threshold: defaultdict(
            lambda: {
                "total": 0,
                "covered": 0,
                "correct": 0,
            }
        )
        for threshold in THRESHOLDS
    }

    print("\n开始严格复现版阈值消融……")

    for index, image_name in enumerate(image_names, start=1):
        image_path = TEST_IMAGE_DIR / image_name
        true_class = infer_true_class(
            image_path,
            sorted_classes,
        )

        if true_class is None:
            for threshold in THRESHOLDS:
                stats_by_threshold[threshold][
                    "unknown_true_class"
                ] += 1
                rows_by_threshold[threshold].append(
                    {
                        "filename": image_name,
                        "covered": False,
                        "correct": False,
                        "status": "unknown_true_class",
                    }
                )
            continue

        for threshold in THRESHOLDS:
            class_stats[threshold][true_class]["total"] += 1

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            for threshold in THRESHOLDS:
                stats_by_threshold[threshold][
                    "image_read_failed"
                ] += 1
                rows_by_threshold[threshold].append(
                    {
                        "filename": image_name,
                        "true_class": true_class,
                        "covered": False,
                        "correct": False,
                        "status": "image_read_failed",
                    }
                )
            continue

        height, width = image_bgr.shape[:2]

        try:
            # 与 crop_pred_final.py 完全相同：
            # 不显式传 conf 和 imgsz，使用Ultralytics默认预测设置。
            result = detector(
                str(image_path),
                verbose=False,
            )[0]
            boxes = result.boxes
        except Exception as exc:
            for threshold in THRESHOLDS:
                stats_by_threshold[threshold][
                    "detector_failed"
                ] += 1
                rows_by_threshold[threshold].append(
                    {
                        "filename": image_name,
                        "true_class": true_class,
                        "covered": False,
                        "correct": False,
                        "status": "detector_failed",
                        "error": str(exc),
                    }
                )
            continue

        if boxes is None or len(boxes) == 0:
            best_box = None
            detector_confidence = None
            detector_pred_class = ""
        else:
            # 与原脚本完全相同
            best_box = max(
                boxes,
                key=lambda box: box.conf[0],
            )
            detector_confidence = float(
                best_box.conf[0]
            )
            detector_class_id = int(best_box.cls[0])
            detector_pred_class = str(
                detector.names[detector_class_id]
            )

        for threshold in THRESHOLDS:
            if (
                best_box is None
                or detector_confidence is None
                or detector_confidence < threshold
            ):
                rows_by_threshold[threshold].append(
                    {
                        "filename": image_name,
                        "true_class": true_class,
                        "detector_pred_class": detector_pred_class,
                        "detector_confidence": (
                            ""
                            if detector_confidence is None
                            else detector_confidence
                        ),
                        "covered": False,
                        "correct": False,
                        "status": "missed_or_below_threshold",
                    }
                )
                continue

            # 与 crop_pred_final.py 完全相同：
            # 先 map(int, ...) 截断原始xyxy浮点坐标
            x1, y1, x2, y2 = map(
                int,
                best_box.xyxy[0].tolist(),
            )

            box_width = x2 - x1
            box_height = y2 - y1

            # 与原脚本完全相同：使用 int() 截断扩展后坐标
            x1 = max(
                0,
                int(x1 - box_width * EXPAND_RATIO),
            )
            y1 = max(
                0,
                int(y1 - box_height * EXPAND_RATIO),
            )
            x2 = min(
                width,
                int(x2 + box_width * EXPAND_RATIO),
            )
            y2 = min(
                height,
                int(y2 + box_height * EXPAND_RATIO),
            )

            roi = image_bgr[y1:y2, x1:x2]
            if roi.size == 0:
                stats_by_threshold[threshold][
                    "empty_roi"
                ] += 1
                rows_by_threshold[threshold].append(
                    {
                        "filename": image_name,
                        "true_class": true_class,
                        "detector_pred_class": detector_pred_class,
                        "detector_confidence": detector_confidence,
                        "covered": False,
                        "correct": False,
                        "status": "empty_roi",
                    }
                )
                continue

            base_name = os.path.splitext(image_name)[0]
            save_name = (
                f"{base_name}_pred_{detector_pred_class}"
                f"_conf{detector_confidence:.2f}.jpg"
            )
            saved_roi_path = (
                crop_dirs[threshold] / save_name
            )

            save_success = cv2.imwrite(
                str(saved_roi_path),
                roi,
            )
            if not save_success:
                stats_by_threshold[threshold][
                    "save_failed"
                ] += 1
                rows_by_threshold[threshold].append(
                    {
                        "filename": image_name,
                        "true_class": true_class,
                        "detector_pred_class": detector_pred_class,
                        "detector_confidence": detector_confidence,
                        "covered": False,
                        "correct": False,
                        "status": "save_failed",
                    }
                )
                continue

            try:
                (
                    classifier_pred_class,
                    classifier_confidence,
                ) = classify_saved_jpg(
                    saved_roi_path,
                    classifier,
                    transform,
                    class_names,
                    device,
                )
            except Exception as exc:
                stats_by_threshold[threshold][
                    "classifier_failed"
                ] += 1
                rows_by_threshold[threshold].append(
                    {
                        "filename": image_name,
                        "saved_roi_filename": save_name,
                        "true_class": true_class,
                        "detector_pred_class": detector_pred_class,
                        "detector_confidence": detector_confidence,
                        "covered": False,
                        "correct": False,
                        "status": "classifier_failed",
                        "error": str(exc),
                    }
                )
                continue

            is_correct = (
                classifier_pred_class == true_class
            )

            stats_by_threshold[threshold][
                "covered"
            ] += 1
            stats_by_threshold[threshold][
                "correct"
            ] += int(is_correct)

            class_stats[threshold][true_class][
                "covered"
            ] += 1
            class_stats[threshold][true_class][
                "correct"
            ] += int(is_correct)

            rows_by_threshold[threshold].append(
                {
                    "filename": image_name,
                    "saved_roi_filename": save_name,
                    "true_class": true_class,
                    "detector_pred_class": detector_pred_class,
                    "detector_confidence": detector_confidence,
                    "classifier_pred_class": (
                        classifier_pred_class
                    ),
                    "classifier_confidence": (
                        classifier_confidence
                    ),
                    "covered": True,
                    "correct": is_correct,
                    "status": (
                        "correct"
                        if is_correct
                        else "misclassified"
                    ),
                }
            )

        if index % 100 == 0:
            print(f"已处理：{index}/{total_images}")

    summary_rows: list[dict[str, object]] = []
    summary_json = {
        "experiment": (
            "confidence_threshold_ablation_"
            "exact_crop_pred_final"
        ),
        "crop_logic": (
            "integer_xyxy_then_15pct_expand_"
            "save_jpg_then_reload"
        ),
        "yolo_model_path": str(YOLO_MODEL_PATH),
        "classifier_model_path": str(
            CLASSIFIER_MODEL_PATH
        ),
        "test_image_dir": str(TEST_IMAGE_DIR),
        "total_images": total_images,
        "roi_expansion": EXPAND_RATIO,
        "thresholds": {},
    }

    for threshold in THRESHOLDS:
        stats = stats_by_threshold[threshold]

        covered = stats["covered"]
        correct = stats["correct"]
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

        row = {
            "confidence_threshold": threshold,
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
            "unknown_true_class": stats[
                "unknown_true_class"
            ],
            "image_read_failed": stats[
                "image_read_failed"
            ],
            "detector_failed": stats[
                "detector_failed"
            ],
            "empty_roi": stats["empty_roi"],
            "save_failed": stats["save_failed"],
            "classifier_failed": stats[
                "classifier_failed"
            ],
        }
        summary_rows.append(row)

        summary_json["thresholds"][
            f"{threshold:.2f}"
        ] = row

        prediction_path = (
            RESULTS_DIR
            / (
                "predictions_"
                + threshold_folder_name(threshold)
                + ".csv"
            )
        )
        write_predictions(
            prediction_path,
            rows_by_threshold[threshold],
        )

        classwise_path = (
            RESULTS_DIR
            / (
                "classwise_"
                + threshold_folder_name(threshold)
                + ".csv"
            )
        )
        with classwise_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "class_name",
                    "total_images",
                    "covered_images",
                    "correct_images",
                    "coverage",
                    "conditional_accuracy",
                    "end_to_end_accuracy",
                ]
            )

            for class_name in class_names:
                class_item = class_stats[threshold][
                    class_name
                ]
                class_total = class_item["total"]
                class_covered = class_item["covered"]
                class_correct = class_item["correct"]

                writer.writerow(
                    [
                        class_name,
                        class_total,
                        class_covered,
                        class_correct,
                        (
                            class_covered / class_total
                            if class_total
                            else 0.0
                        ),
                        (
                            class_correct / class_covered
                            if class_covered
                            else 0.0
                        ),
                        (
                            class_correct / class_total
                            if class_total
                            else 0.0
                        ),
                    ]
                )

    summary_csv_path = RESULTS_DIR / "threshold_summary.csv"
    with summary_csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json_path = RESULTS_DIR / "summary.json"
    summary_json_path.write_text(
        json.dumps(
            summary_json,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n✅ 严格复现版阈值消融完成")
    print(
        "阈值 | 覆盖数 | 正确数 | 覆盖率 | "
        "条件准确率 | 完整端到端准确率"
    )

    for row in summary_rows:
        print(
            f"{row['confidence_threshold']:.2f} | "
            f"{row['covered_images']} | "
            f"{row['correct_images']} | "
            f"{row['detection_coverage'] * 100:.2f}% | "
            f"{row['conditional_classification_accuracy'] * 100:.2f}% | "
            f"{row['complete_end_to_end_accuracy'] * 100:.2f}%"
        )

    print("\n结果目录：", RESULTS_DIR)
    print("汇总CSV：", summary_csv_path)
    print("汇总JSON：", summary_json_path)
    print(
        "\n关键校验：0.50阈值应为 "
        "664个有效ROI、604个正确、88.82%端到端准确率。"
    )


if __name__ == "__main__":
    main()
