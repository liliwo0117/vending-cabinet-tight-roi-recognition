# ============================================================================
# FINAL paired test, tau=0.01, rho=0 (paper Fig. 8).
# Repository name : s32_final_test_tight_roi.py
# Original name   : 最终公平测试_阈值0.01_紧致ROI.py
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
# 最终公平测试：
# - 参数由验证集选定：检测阈值0.01、ROI扩展0%
# - 同一批检测结果同时评价：
#   1) 单阶段YOLO11类别
#   2) YOLO11 + 紧致ROI + ConvNeXt V2-Atto
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
    / "final_test_conf_0_01_tight_roi"
)
ROI_DIR = RESULTS_DIR / "saved_tight_rois"

CONF_THRESHOLD = 0.01
ROI_EXPANSION = 0.0
EXPECTED_TEST_IMAGE_TOTAL = 680
IMAGE_EXTENSIONS = {".jpg"}


def load_state_dict_compat(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def infer_true_class(
    image_name: str,
    sorted_classes: list[str],
) -> str | None:
    name_clean = Path(image_name).stem.split("(")[0]
    return next(
        (
            class_name
            for class_name in sorted_classes
            if name_clean.startswith(class_name)
        ),
        None,
    )


def classify_saved_roi(
    roi_path: Path,
    classifier,
    transform,
    class_names: list[str],
    device: torch.device,
) -> tuple[str, float]:
    image = Image.open(roi_path).convert("RGB")
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
    print("测试集目录：", TEST_IMAGE_DIR)
    print("结果目录：", RESULTS_DIR)
    print("验证集锁定检测阈值：", CONF_THRESHOLD)
    print("验证集锁定ROI扩展：0%")

    for path, description in [
        (YOLO_MODEL_PATH, "YOLO权重"),
        (CLASSIFIER_MODEL_PATH, "分类器权重"),
        (TRAIN_DATA_DIR, "分类训练目录"),
        (TEST_IMAGE_DIR, "测试图片目录"),
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

    rows: list[dict[str, object]] = []

    class_stats = defaultdict(
        lambda: {
            "total": 0,
            "covered": 0,
            "yolo_correct": 0,
            "two_stage_correct": 0,
        }
    )

    covered = 0
    yolo_correct = 0
    two_stage_correct = 0

    both_correct = 0
    both_wrong = 0
    yolo_only_correct = 0
    two_stage_only_correct = 0

    unknown_true_class = 0
    image_read_failed = 0
    detector_failed = 0
    empty_roi = 0
    save_failed = 0
    classifier_failed = 0

    print("\n开始最终公平测试……")

    for index, image_name in enumerate(image_names, start=1):
        image_path = TEST_IMAGE_DIR / image_name
        true_class = infer_true_class(image_name, sorted_classes)

        row: dict[str, object] = {
            "filename": image_name,
            "true_class": true_class or "",
            "detector_confidence": "",
            "yolo_pred_class": "",
            "yolo_correct": False,
            "classifier_pred_class": "",
            "classifier_confidence": "",
            "two_stage_correct": False,
            "covered": False,
            "status": "",
            "error": "",
        }

        if true_class is None:
            unknown_true_class += 1
            row["status"] = "unknown_true_class"
            rows.append(row)
            continue

        class_stats[true_class]["total"] += 1

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            image_read_failed += 1
            row["status"] = "image_read_failed"
            rows.append(row)
            continue

        height, width = image_bgr.shape[:2]

        try:
            result = detector.predict(
                source=str(image_path),
                conf=CONF_THRESHOLD,
                device=yolo_device,
                save=False,
                verbose=False,
            )[0]
            boxes = result.boxes
        except Exception as exc:
            detector_failed += 1
            row["status"] = "detector_failed"
            row["error"] = str(exc)
            rows.append(row)
            continue

        if boxes is None or len(boxes) == 0:
            row["status"] = "missed_or_below_threshold"
            rows.append(row)
            continue

        best_box = max(boxes, key=lambda box: box.conf[0])
        detector_confidence = float(best_box.conf[0])
        detector_class_id = int(best_box.cls[0])
        yolo_pred_class = str(detector.names[detector_class_id])
        yolo_is_correct = yolo_pred_class == true_class

        row["detector_confidence"] = detector_confidence
        row["yolo_pred_class"] = yolo_pred_class
        row["yolo_correct"] = yolo_is_correct
        row["covered"] = True

        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        roi = image_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            empty_roi += 1
            row["covered"] = False
            row["status"] = "empty_roi"
            rows.append(row)
            continue

        roi_name = (
            f"{Path(image_name).stem}_pred_{yolo_pred_class}"
            f"_conf{detector_confidence:.4f}_tight.jpg"
        )
        roi_path = ROI_DIR / roi_name

        if not cv2.imwrite(str(roi_path), roi):
            save_failed += 1
            row["covered"] = False
            row["status"] = "save_failed"
            rows.append(row)
            continue

        try:
            classifier_pred_class, classifier_confidence = (
                classify_saved_roi(
                    roi_path,
                    classifier,
                    transform,
                    class_names,
                    device,
                )
            )
        except Exception as exc:
            classifier_failed += 1
            row["covered"] = False
            row["status"] = "classifier_failed"
            row["error"] = str(exc)
            rows.append(row)
            continue

        two_stage_is_correct = classifier_pred_class == true_class

        row["classifier_pred_class"] = classifier_pred_class
        row["classifier_confidence"] = classifier_confidence
        row["two_stage_correct"] = two_stage_is_correct
        row["status"] = "processed"

        covered += 1
        yolo_correct += int(yolo_is_correct)
        two_stage_correct += int(two_stage_is_correct)

        class_stats[true_class]["covered"] += 1
        class_stats[true_class]["yolo_correct"] += int(yolo_is_correct)
        class_stats[true_class]["two_stage_correct"] += int(
            two_stage_is_correct
        )

        if yolo_is_correct and two_stage_is_correct:
            both_correct += 1
        elif (not yolo_is_correct) and (not two_stage_is_correct):
            both_wrong += 1
        elif yolo_is_correct and (not two_stage_is_correct):
            yolo_only_correct += 1
        else:
            two_stage_only_correct += 1

        rows.append(row)

        if index % 100 == 0:
            print(f"已处理：{index}/{total_images}")

    missed = total_images - covered
    detection_coverage = covered / total_images if total_images else 0.0

    yolo_conditional_accuracy = (
        yolo_correct / covered if covered else 0.0
    )
    yolo_end_to_end_accuracy = (
        yolo_correct / total_images if total_images else 0.0
    )

    two_stage_conditional_accuracy = (
        two_stage_correct / covered if covered else 0.0
    )
    two_stage_end_to_end_accuracy = (
        two_stage_correct / total_images if total_images else 0.0
    )

    predictions_path = RESULTS_DIR / "paired_predictions.csv"
    prediction_fields = [
        "filename",
        "true_class",
        "detector_confidence",
        "yolo_pred_class",
        "yolo_correct",
        "classifier_pred_class",
        "classifier_confidence",
        "two_stage_correct",
        "covered",
        "status",
        "error",
    ]
    with predictions_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=prediction_fields)
        writer.writeheader()
        writer.writerows(rows)

    classwise_path = RESULTS_DIR / "classwise_comparison.csv"
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
                "yolo_correct",
                "yolo_conditional_accuracy",
                "yolo_end_to_end_accuracy",
                "two_stage_correct",
                "two_stage_conditional_accuracy",
                "two_stage_end_to_end_accuracy",
                "two_stage_minus_yolo_correct",
            ]
        )

        for class_name in class_names:
            item = class_stats[class_name]
            class_total = item["total"]
            class_covered = item["covered"]
            class_yolo_correct = item["yolo_correct"]
            class_two_stage_correct = item["two_stage_correct"]

            writer.writerow(
                [
                    class_name,
                    class_total,
                    class_covered,
                    class_yolo_correct,
                    (
                        class_yolo_correct / class_covered
                        if class_covered
                        else 0.0
                    ),
                    (
                        class_yolo_correct / class_total
                        if class_total
                        else 0.0
                    ),
                    class_two_stage_correct,
                    (
                        class_two_stage_correct / class_covered
                        if class_covered
                        else 0.0
                    ),
                    (
                        class_two_stage_correct / class_total
                        if class_total
                        else 0.0
                    ),
                    class_two_stage_correct - class_yolo_correct,
                ]
            )

    summary = {
        "experiment": "final_paired_test_after_validation_selection",
        "selection_split": "validation",
        "locked_confidence_threshold": CONF_THRESHOLD,
        "locked_roi_expansion_ratio": ROI_EXPANSION,
        "total_images": total_images,
        "covered_images": covered,
        "missed_or_below_threshold": missed,
        "detection_coverage": detection_coverage,
        "single_stage_yolo11": {
            "correct_images": yolo_correct,
            "conditional_accuracy": yolo_conditional_accuracy,
            "complete_end_to_end_accuracy": yolo_end_to_end_accuracy,
        },
        "two_stage_yolo11_convnextv2": {
            "correct_images": two_stage_correct,
            "conditional_accuracy": two_stage_conditional_accuracy,
            "complete_end_to_end_accuracy": two_stage_end_to_end_accuracy,
        },
        "improvement": {
            "correct_images": two_stage_correct - yolo_correct,
            "conditional_accuracy_points": (
                two_stage_conditional_accuracy
                - yolo_conditional_accuracy
            ),
            "complete_end_to_end_accuracy_points": (
                two_stage_end_to_end_accuracy
                - yolo_end_to_end_accuracy
            ),
        },
        "paired_outcomes": {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "yolo_only_correct": yolo_only_correct,
            "two_stage_only_correct": two_stage_only_correct,
        },
        "errors": {
            "unknown_true_class": unknown_true_class,
            "image_read_failed": image_read_failed,
            "detector_failed": detector_failed,
            "empty_roi": empty_roi,
            "save_failed": save_failed,
            "classifier_failed": classifier_failed,
        },
        "yolo_model_path": str(YOLO_MODEL_PATH),
        "classifier_model_path": str(CLASSIFIER_MODEL_PATH),
        "test_image_dir": str(TEST_IMAGE_DIR),
    }

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n✅ 最终公平测试完成")
    print(f"测试图像数：{total_images}")
    print(f"有效检测/ROI数：{covered}")
    print(f"漏检或低置信度数：{missed}")
    print(f"检测覆盖率：{detection_coverage * 100:.2f}%")

    print("\n单阶段YOLO11：")
    print(f"正确数：{yolo_correct}")
    print(f"条件准确率：{yolo_conditional_accuracy * 100:.2f}%")
    print(f"端到端准确率：{yolo_end_to_end_accuracy * 100:.2f}%")

    print("\n两阶段YOLO11 + ConvNeXt V2：")
    print(f"正确数：{two_stage_correct}")
    print(
        f"条件准确率："
        f"{two_stage_conditional_accuracy * 100:.2f}%"
    )
    print(
        f"端到端准确率："
        f"{two_stage_end_to_end_accuracy * 100:.2f}%"
    )

    print("\n两阶段相对单阶段：")
    print(f"净增加正确数：{two_stage_correct - yolo_correct}")
    print(
        f"条件准确率提升："
        f"{(two_stage_conditional_accuracy - yolo_conditional_accuracy) * 100:.2f}"
        " 个百分点"
    )
    print(
        f"端到端准确率提升："
        f"{(two_stage_end_to_end_accuracy - yolo_end_to_end_accuracy) * 100:.2f}"
        " 个百分点"
    )

    print("\n逐图配对：")
    print(f"两者都正确：{both_correct}")
    print(f"两者都错误：{both_wrong}")
    print(f"仅YOLO正确：{yolo_only_correct}")
    print(f"仅两阶段正确：{two_stage_only_correct}")

    print("\n异常统计：", summary["errors"])
    print("结果目录：", RESULTS_DIR)
    print("汇总文件：", summary_path)
    print("逐样本结果：", predictions_path)
    print("分类别比较：", classwise_path)


if __name__ == "__main__":
    main()
