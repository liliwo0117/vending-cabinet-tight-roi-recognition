# ============================================================================
# Fair single-stage baseline at confidence 0.30.
# Repository name : s31_baseline_single_stage_conf030.py
# Original name   : 单阶段YOLO11公平基线_阈值0.30.py
# Paper           : Task-Decoupled Fine-Grained Product Recognition Using
#                   Tight Predicted ROIs for Automated Vending Cabinets
# Note            : algorithm unchanged; comments kept in the original Chinese.
# ============================================================================
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO


# 将本脚本放在项目根目录运行
PROJECT_ROOT = Path(__file__).resolve().parent

YOLO_MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "train_8cls_gpu2"
    / "weights"
    / "best.pt"
)

TEST_IMAGE_DIR = PROJECT_ROOT / "data" / "test"

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "single_stage_yolo11_conf_0_30"
)

CONF_THRESHOLD = 0.30
EXPECTED_TEST_IMAGE_TOTAL = 680
IMAGE_EXTENSIONS = {".jpg"}

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


def infer_true_class(image_name: str) -> str | None:
    name_clean = Path(image_name).stem.split("(")[0]
    for class_name in sorted(
        CLASS_NAMES,
        key=len,
        reverse=True,
    ):
        if name_clean.startswith(class_name):
            return class_name
    return None


def main() -> None:
    print("YOLO权重：", YOLO_MODEL_PATH)
    print("测试集目录：", TEST_IMAGE_DIR)
    print("结果目录：", RESULTS_DIR)
    print("检测置信度阈值：", CONF_THRESHOLD)

    if not YOLO_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"找不到YOLO权重：{YOLO_MODEL_PATH}"
        )

    if not TEST_IMAGE_DIR.is_dir():
        raise FileNotFoundError(
            f"找不到测试集目录：{TEST_IMAGE_DIR}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(YOLO_MODEL_PATH))

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
            "correct": 0,
        }
    )

    covered_images = 0
    correct_images = 0
    unknown_true_class = 0
    inference_failed = 0

    print("\n开始单阶段YOLO11公平基线评价……")

    for index, image_name in enumerate(
        image_names,
        start=1,
    ):
        image_path = TEST_IMAGE_DIR / image_name
        true_class = infer_true_class(image_name)

        if true_class is None:
            unknown_true_class += 1
            rows.append(
                {
                    "filename": image_name,
                    "true_class": "",
                    "pred_class": "",
                    "confidence": "",
                    "covered": False,
                    "correct": False,
                    "status": "unknown_true_class",
                    "error": "",
                }
            )
            continue

        class_stats[true_class]["total"] += 1

        try:
            # 与正式裁剪脚本一致：使用Ultralytics默认预测设置，
            # 然后从全部检测框中选择最高置信度框。
            result = model(
                str(image_path),
                verbose=False,
            )[0]
            boxes = result.boxes
        except Exception as exc:
            inference_failed += 1
            rows.append(
                {
                    "filename": image_name,
                    "true_class": true_class,
                    "pred_class": "",
                    "confidence": "",
                    "covered": False,
                    "correct": False,
                    "status": "inference_failed",
                    "error": str(exc),
                }
            )
            continue

        if boxes is None or len(boxes) == 0:
            rows.append(
                {
                    "filename": image_name,
                    "true_class": true_class,
                    "pred_class": "",
                    "confidence": "",
                    "covered": False,
                    "correct": False,
                    "status": "missed_or_below_threshold",
                    "error": "",
                }
            )
            continue

        best_box = max(
            boxes,
            key=lambda box: box.conf[0],
        )

        confidence = float(best_box.conf[0])

        if confidence < CONF_THRESHOLD:
            rows.append(
                {
                    "filename": image_name,
                    "true_class": true_class,
                    "pred_class": "",
                    "confidence": confidence,
                    "covered": False,
                    "correct": False,
                    "status": "missed_or_below_threshold",
                    "error": "",
                }
            )
            continue

        pred_class_id = int(best_box.cls[0])
        pred_class = str(model.names[pred_class_id])

        is_correct = pred_class == true_class

        covered_images += 1
        correct_images += int(is_correct)

        class_stats[true_class]["covered"] += 1
        class_stats[true_class]["correct"] += int(
            is_correct
        )

        rows.append(
            {
                "filename": image_name,
                "true_class": true_class,
                "pred_class": pred_class,
                "confidence": confidence,
                "covered": True,
                "correct": is_correct,
                "status": (
                    "correct"
                    if is_correct
                    else "misclassified"
                ),
                "error": "",
            }
        )

        if index % 100 == 0:
            print(f"已处理：{index}/{total_images}")

    missed_images = total_images - covered_images
    detection_coverage = (
        covered_images / total_images
        if total_images
        else 0.0
    )
    conditional_accuracy = (
        correct_images / covered_images
        if covered_images
        else 0.0
    )
    end_to_end_accuracy = (
        correct_images / total_images
        if total_images
        else 0.0
    )

    predictions_path = RESULTS_DIR / "predictions.csv"
    fieldnames = [
        "filename",
        "true_class",
        "pred_class",
        "confidence",
        "covered",
        "correct",
        "status",
        "error",
    ]

    with predictions_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    classwise_path = RESULTS_DIR / "classwise_metrics.csv"
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
            item = class_stats[class_name]
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

    summary = {
        "method": "single_stage_yolo11",
        "confidence_threshold": CONF_THRESHOLD,
        "model_path": str(YOLO_MODEL_PATH),
        "test_dir": str(TEST_IMAGE_DIR),
        "total_images": total_images,
        "covered_images": covered_images,
        "missed_or_below_threshold": missed_images,
        "correct_images": correct_images,
        "detection_coverage": detection_coverage,
        "conditional_classification_accuracy": (
            conditional_accuracy
        ),
        "complete_end_to_end_accuracy": (
            end_to_end_accuracy
        ),
        "unknown_true_class": unknown_true_class,
        "inference_failed": inference_failed,
        "class_names": CLASS_NAMES,
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

    print("\n✅ 单阶段YOLO11公平基线评价完成")
    print(f"原始测试图像数：{total_images}")
    print(f"有效检测图像数：{covered_images}")
    print(f"漏检或低置信度数：{missed_images}")
    print(f"类别预测正确数：{correct_images}")
    print(f"检测覆盖率：{detection_coverage * 100:.2f}%")
    print(
        f"条件类别准确率："
        f"{conditional_accuracy * 100:.2f}%"
    )
    print(
        f"完整端到端准确率："
        f"{end_to_end_accuracy * 100:.2f}%"
    )
    print(f"无法解析真实类别：{unknown_true_class}")
    print(f"推理失败：{inference_failed}")
    print("结果目录：", RESULTS_DIR)
    print("汇总文件：", summary_path)
    print("逐样本结果：", predictions_path)
    print("分类别结果：", classwise_path)
    print(
        "\n关键校验：有效检测数应为677，"
        "与两阶段阈值0.30实验一致。"
    )


if __name__ == "__main__":
    main()
