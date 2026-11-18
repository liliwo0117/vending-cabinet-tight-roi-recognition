# ============================================================================
# Confidence ablation with the ImageNet-initialised classifier.
# Repository name : s21_ablation_confidence_imagenet.py
# Original name   : 检测置信度阈值消融_官方ImageNet.py
# Paper           : Task-Decoupled Fine-Grained Product Recognition Using
#                   Tight Predicted ROIs for Automated Vending Cabinets
# Note            : algorithm unchanged; comments kept in the original Chinese.
# ============================================================================
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import timm
import torch
from PIL import Image
from torchvision import datasets, transforms
from ultralytics import YOLO


# ============================================================
# 将本脚本放在项目根目录：
# 
#(repository root)
# 实验目的：
# 在同一测试集、同一YOLO权重、同一15% ROI扩展、
# 同一ConvNeXt V2-Atto分类器下，仅改变检测置信度阈值。
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
    / "confidence_threshold_ablation_imagenet_official"
)

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]
ROI_EXPANSION = 0.15
YOLO_IMAGE_SIZE = 640
TEST_IMAGE_TOTAL = 680

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


def expand_box(
    xyxy: list[float],
    image_width: int,
    image_height: int,
    expansion: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy

    box_width = x2 - x1
    box_height = y2 - y1

    expanded_x1 = max(
        0,
        int(round(x1 - expansion * box_width)),
    )
    expanded_y1 = max(
        0,
        int(round(y1 - expansion * box_height)),
    )
    expanded_x2 = min(
        image_width,
        int(round(x2 + expansion * box_width)),
    )
    expanded_y2 = min(
        image_height,
        int(round(y2 + expansion * box_height)),
    )

    return (
        expanded_x1,
        expanded_y1,
        expanded_x2,
        expanded_y2,
    )


def classify_roi(
    roi_bgr,
    classifier,
    transform,
    class_names: list[str],
    device: torch.device,
) -> tuple[str, float]:
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(roi_rgb)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = classifier(tensor)
        probabilities = torch.softmax(logits, dim=1)
        pred_index = int(probabilities.argmax(dim=1).item())
        confidence = float(probabilities[0, pred_index].item())

    return class_names[pred_index], confidence


def write_predictions_csv(
    csv_path: Path,
    rows: list[dict[str, object]],
) -> None:
    fieldnames = [
        "filename",
        "true_class",
        "det_confidence",
        "pred_class",
        "classifier_confidence",
        "covered",
        "correct",
        "status",
        "error",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field_name: row.get(field_name, "")
                    for field_name in fieldnames
                }
            )


def main() -> None:
    print("YOLO权重：", YOLO_MODEL_PATH)
    print("分类器权重：", CLASSIFIER_MODEL_PATH)
    print("测试集目录：", TEST_IMAGE_DIR)
    print("结果目录：", RESULTS_DIR)
    print("阈值：", THRESHOLDS)
    print(f"ROI扩展比例：{ROI_EXPANSION:.0%}")

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
            f"找不到测试图像目录：{TEST_IMAGE_DIR}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
    yolo_device: int | str = (
        0 if torch.cuda.is_available() else "cpu"
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
        for path in TEST_IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if len(image_paths) != TEST_IMAGE_TOTAL:
        print(
            f"⚠ 测试图片数量为 {len(image_paths)}，"
            f"不是预期的 {TEST_IMAGE_TOTAL}。"
        )

    threshold_rows: dict[float, list[dict[str, object]]] = {
        threshold: []
        for threshold in THRESHOLDS
    }

    threshold_stats = {
        threshold: {
            "covered": 0,
            "correct": 0,
            "failed": 0,
            "unknown_true_class": 0,
            "empty_roi": 0,
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

    # 只进行一次YOLO推理。
    # 使用最低阈值0.30得到候选框，再按各阈值过滤最高置信度框。
    minimum_threshold = min(THRESHOLDS)

    print("\n开始检测阈值消融实验……")

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        true_class = infer_true_class(
            image_path,
            sorted_classes,
        )

        if true_class is None:
            for threshold in THRESHOLDS:
                threshold_stats[threshold][
                    "unknown_true_class"
                ] += 1
                threshold_rows[threshold].append(
                    {
                        "filename": image_path.name,
                        "true_class": "",
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
                threshold_stats[threshold]["failed"] += 1
                threshold_rows[threshold].append(
                    {
                        "filename": image_path.name,
                        "true_class": true_class,
                        "covered": False,
                        "correct": False,
                        "status": "image_read_failed",
                    }
                )
            continue

        try:
            result = detector.predict(
                source=str(image_path),
                imgsz=YOLO_IMAGE_SIZE,
                conf=minimum_threshold,
                device=yolo_device,
                save=False,
                verbose=False,
            )[0]
        except Exception as exc:
            for threshold in THRESHOLDS:
                threshold_stats[threshold]["failed"] += 1
                threshold_rows[threshold].append(
                    {
                        "filename": image_path.name,
                        "true_class": true_class,
                        "covered": False,
                        "correct": False,
                        "status": "detector_failed",
                        "error": str(exc),
                    }
                )
            continue

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            best_det_confidence = None
            best_xyxy = None
        else:
            confidences = boxes.conf.detach().cpu()
            best_index = int(
                torch.argmax(confidences).item()
            )
            best_det_confidence = float(
                confidences[best_index].item()
            )
            best_xyxy = (
                boxes.xyxy[best_index]
                .detach()
                .cpu()
                .tolist()
            )

        for threshold in THRESHOLDS:
            if (
                best_det_confidence is None
                or best_det_confidence < threshold
                or best_xyxy is None
            ):
                threshold_rows[threshold].append(
                    {
                        "filename": image_path.name,
                        "true_class": true_class,
                        "det_confidence": (
                            ""
                            if best_det_confidence is None
                            else best_det_confidence
                        ),
                        "covered": False,
                        "correct": False,
                        "status": "missed_or_below_threshold",
                    }
                )
                continue

            image_height, image_width = image_bgr.shape[:2]
            x1, y1, x2, y2 = expand_box(
                best_xyxy,
                image_width,
                image_height,
                ROI_EXPANSION,
            )

            roi = image_bgr[y1:y2, x1:x2]
            if roi.size == 0:
                threshold_stats[threshold][
                    "empty_roi"
                ] += 1
                threshold_rows[threshold].append(
                    {
                        "filename": image_path.name,
                        "true_class": true_class,
                        "det_confidence": best_det_confidence,
                        "covered": False,
                        "correct": False,
                        "status": "empty_roi",
                    }
                )
                continue

            try:
                pred_class, classifier_confidence = (
                    classify_roi(
                        roi,
                        classifier,
                        transform,
                        class_names,
                        device,
                    )
                )
            except Exception as exc:
                threshold_stats[threshold]["failed"] += 1
                threshold_rows[threshold].append(
                    {
                        "filename": image_path.name,
                        "true_class": true_class,
                        "det_confidence": best_det_confidence,
                        "covered": False,
                        "correct": False,
                        "status": "classifier_failed",
                        "error": str(exc),
                    }
                )
                continue

            is_correct = pred_class == true_class

            threshold_stats[threshold]["covered"] += 1
            threshold_stats[threshold]["correct"] += int(
                is_correct
            )

            class_stats[threshold][true_class][
                "covered"
            ] += 1
            class_stats[threshold][true_class][
                "correct"
            ] += int(is_correct)

            threshold_rows[threshold].append(
                {
                    "filename": image_path.name,
                    "true_class": true_class,
                    "det_confidence": best_det_confidence,
                    "pred_class": pred_class,
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
            print(
                f"已处理：{index}/{len(image_paths)}"
            )

    summary_rows = []
    summary_json = {
        "experiment": "confidence_threshold_ablation",
        "yolo_model_path": str(YOLO_MODEL_PATH),
        "classifier_model_path": str(
            CLASSIFIER_MODEL_PATH
        ),
        "test_image_dir": str(TEST_IMAGE_DIR),
        "test_image_total": len(image_paths),
        "roi_expansion": ROI_EXPANSION,
        "yolo_image_size": YOLO_IMAGE_SIZE,
        "thresholds": {},
    }

    for threshold in THRESHOLDS:
        stats = threshold_stats[threshold]
        covered = stats["covered"]
        correct = stats["correct"]
        total = len(image_paths)

        missed = total - covered
        coverage = covered / total if total else 0.0
        conditional_accuracy = (
            correct / covered if covered else 0.0
        )
        end_to_end_accuracy = (
            correct / total if total else 0.0
        )

        threshold_name = f"{threshold:.2f}".replace(
            ".",
            "_",
        )

        predictions_path = (
            RESULTS_DIR
            / f"predictions_conf_{threshold_name}.csv"
        )
        write_predictions_csv(
            predictions_path,
            threshold_rows[threshold],
        )

        classwise_path = (
            RESULTS_DIR
            / f"classwise_conf_{threshold_name}.csv"
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
                item = class_stats[threshold][
                    class_name
                ]
                class_total = item["total"]
                class_covered = item["covered"]
                class_correct = item["correct"]

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

        row = {
            "confidence_threshold": threshold,
            "total_images": total,
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
            "empty_roi": stats["empty_roi"],
            "failed": stats["failed"],
        }
        summary_rows.append(row)

        summary_json["thresholds"][f"{threshold:.2f}"] = row

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

    print("\n✅ 检测置信度阈值消融完成")
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
        "\n重要校验：0.50阈值应接近当前正式结果 "
        "664个有效ROI、604个正确、88.82%端到端准确率。"
    )


if __name__ == "__main__":
    main()
