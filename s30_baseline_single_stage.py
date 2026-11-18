# ============================================================================
# Run the single-stage YOLO11m baseline.
# Repository name : s30_baseline_single_stage.py
# Original name   : 运行单阶段YOLO11基线.py
# Paper           : Task-Decoupled Fine-Grained Product Recognition Using
#                   Tight Predicted ROIs for Automated Vending Cabinets
# Note            : algorithm unchanged; comments kept in the original Chinese.
# ============================================================================
from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "train_8cls_gpu2"
    / "weights"
    / "best.pt"
)

TEST_DIR = PROJECT_ROOT / "data" / "test"

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "single_stage_yolo11_baseline"
)

MISCLASSIFIED_DIR = RESULT_DIR / "misclassified"
MISSED_DIR = RESULT_DIR / "missed"

CONF_THRESHOLD = 0.50
IMAGE_SIZE = 640

CLASS_NAMES = [
    "hks_large",
    "hks_small",
    "hn_can",
    "jlb_can",
    "kkkl_can",
    "wlj_can",
    "xb",
    "xb_wt",
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def infer_true_class(image_path: Path) -> str | None:
    for class_name in sorted(CLASS_NAMES, key=len, reverse=True):
        if image_path.stem.startswith(class_name):
            return class_name
    return None


def prepare_output_dirs() -> None:
    if RESULT_DIR.exists():
        shutil.rmtree(RESULT_DIR)

    MISCLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)
    MISSED_DIR.mkdir(parents=True, exist_ok=True)


def save_annotated_result(result, output_path: Path) -> None:
    plotted_bgr = result.plot()
    plotted_rgb = plotted_bgr[:, :, ::-1]
    Image.fromarray(plotted_rgb).save(output_path)


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"找不到YOLO11权重：{MODEL_PATH}")

    if not TEST_DIR.is_dir():
        raise FileNotFoundError(f"找不到测试集目录：{TEST_DIR}")

    prepare_output_dirs()

    device: int | str = 0 if torch.cuda.is_available() else "cpu"

    print("YOLO11权重：", MODEL_PATH)
    print("测试集目录：", TEST_DIR)
    print("结果目录：", RESULT_DIR)
    print("置信度阈值：", CONF_THRESHOLD)
    print("运行设备：", device)

    model = YOLO(str(MODEL_PATH))

    image_paths = sorted(
        path
        for path in TEST_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    records: list[dict[str, object]] = []
    class_stats = defaultdict(lambda: {"total": 0, "covered": 0, "correct": 0})

    total_images = 0
    covered_images = 0
    correct_images = 0
    missed_images = 0
    skipped_unknown_class = 0
    failed_images = 0

    for index, image_path in enumerate(image_paths, start=1):
        total_images += 1
        true_class = infer_true_class(image_path)

        if true_class is None:
            skipped_unknown_class += 1
            records.append(
                {
                    "filename": image_path.name,
                    "true_class": "",
                    "pred_class": "",
                    "confidence": "",
                    "num_detections": 0,
                    "covered": False,
                    "correct": False,
                    "status": "unknown_true_class",
                }
            )
            continue

        class_stats[true_class]["total"] += 1

        try:
            result = model.predict(
                source=str(image_path),
                imgsz=IMAGE_SIZE,
                conf=CONF_THRESHOLD,
                device=device,
                save=False,
                verbose=False,
            )[0]
        except Exception as exc:
            failed_images += 1
            records.append(
                {
                    "filename": image_path.name,
                    "true_class": true_class,
                    "pred_class": "",
                    "confidence": "",
                    "num_detections": 0,
                    "covered": False,
                    "correct": False,
                    "status": f"inference_failed: {exc}",
                }
            )
            continue

        boxes = result.boxes
        num_detections = 0 if boxes is None else len(boxes)

        if boxes is None or len(boxes) == 0:
            missed_images += 1
            shutil.copy2(image_path, MISSED_DIR / image_path.name)

            records.append(
                {
                    "filename": image_path.name,
                    "true_class": true_class,
                    "pred_class": "",
                    "confidence": "",
                    "num_detections": 0,
                    "covered": False,
                    "correct": False,
                    "status": "missed_or_below_threshold",
                }
            )
            continue

        confidences = boxes.conf.detach().cpu()
        classes = boxes.cls.detach().cpu().to(torch.int64)
        best_index = int(torch.argmax(confidences).item())

        pred_class_id = int(classes[best_index].item())
        pred_class = str(result.names[pred_class_id])
        confidence = float(confidences[best_index].item())

        covered_images += 1
        class_stats[true_class]["covered"] += 1

        is_correct = pred_class == true_class
        if is_correct:
            correct_images += 1
            class_stats[true_class]["correct"] += 1
            status = "correct"
        else:
            status = "misclassified"
            output_name = (
                f"GT_{true_class}__Pred_{pred_class}__"
                f"Conf_{confidence:.4f}__{image_path.name}"
            )
            save_annotated_result(
                result,
                MISCLASSIFIED_DIR / output_name,
            )

        records.append(
            {
                "filename": image_path.name,
                "true_class": true_class,
                "pred_class": pred_class,
                "confidence": confidence,
                "num_detections": num_detections,
                "covered": True,
                "correct": is_correct,
                "status": status,
            }
        )

        if index % 100 == 0:
            print(f"已处理：{index}/{len(image_paths)}")

    detection_coverage = (
        covered_images / total_images if total_images else 0.0
    )
    conditional_accuracy = (
        correct_images / covered_images if covered_images else 0.0
    )
    end_to_end_accuracy = (
        correct_images / total_images if total_images else 0.0
    )

    predictions_path = RESULT_DIR / "predictions.csv"
    with predictions_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        fieldnames = [
            "filename",
            "true_class",
            "pred_class",
            "confidence",
            "num_detections",
            "covered",
            "correct",
            "status",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    classwise_path = RESULT_DIR / "classwise_metrics.csv"
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

        for class_name in CLASS_NAMES:
            stats = class_stats[class_name]
            total = stats["total"]
            covered = stats["covered"]
            correct = stats["correct"]

            writer.writerow(
                [
                    class_name,
                    total,
                    covered,
                    correct,
                    covered / total if total else 0.0,
                    correct / covered if covered else 0.0,
                    correct / total if total else 0.0,
                ]
            )

    summary = {
        "method": "single_stage_yolo11",
        "model_path": str(MODEL_PATH),
        "test_dir": str(TEST_DIR),
        "confidence_threshold": CONF_THRESHOLD,
        "image_size": IMAGE_SIZE,
        "total_images": total_images,
        "covered_images": covered_images,
        "missed_or_below_threshold": missed_images,
        "correct_images": correct_images,
        "detection_coverage": detection_coverage,
        "conditional_classification_accuracy": conditional_accuracy,
        "complete_end_to_end_accuracy": end_to_end_accuracy,
        "skipped_unknown_class": skipped_unknown_class,
        "failed_images": failed_images,
        "class_names": CLASS_NAMES,
    }

    summary_path = RESULT_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n✅ 单阶段YOLO11基线评价完成")
    print(f"原始测试图像数：{total_images}")
    print(f"有效检测图像数：{covered_images}")
    print(f"漏检或低置信度数：{missed_images}")
    print(f"类别预测正确数：{correct_images}")
    print(f"检测覆盖率：{detection_coverage * 100:.2f}%")
    print(f"条件类别准确率：{conditional_accuracy * 100:.2f}%")
    print(f"完整端到端准确率：{end_to_end_accuracy * 100:.2f}%")
    print(f"无法解析真实类别：{skipped_unknown_class}")
    print(f"推理失败：{failed_images}")
    print("结果目录：", RESULT_DIR)
    print("汇总文件：", summary_path)
    print("逐样本结果：", predictions_path)
    print("分类别结果：", classwise_path)


if __name__ == "__main__":
    main()
