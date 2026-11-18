# ============================================================================
# Wilson CI, exact McNemar test, confusion matrices (paper Fig. 9).
# Repository name : s33_final_statistical_analysis.py
# Original name   : 最终统计分析_置信区间_McNemar_混淆矩阵.py
# Paper           : Task-Decoupled Fine-Grained Product Recognition Using
#                   Tight Predicted ROIs for Automated Vending Cabinets
# Note            : algorithm unchanged; comments kept in the original Chinese.
# ============================================================================
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 将本脚本放在项目根目录运行
PROJECT_ROOT = Path(__file__).resolve().parent

PREDICTIONS_CSV = (
    PROJECT_ROOT
    / "results"
    / "final_test_conf_0_01_tight_roi"
    / "paired_predictions.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_statistical_analysis"
)

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


def wilson_interval(
    correct: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0

    proportion = correct / total
    denominator = 1.0 + z * z / total
    center = (
        proportion
        + z * z / (2.0 * total)
    ) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )

    return max(0.0, center - margin), min(1.0, center + margin)


def exact_mcnemar_p_value(
    yolo_only_correct: int,
    two_stage_only_correct: int,
) -> float:
    discordant = yolo_only_correct + two_stage_only_correct
    if discordant == 0:
        return 1.0

    smaller = min(
        yolo_only_correct,
        two_stage_only_correct,
    )

    cumulative = sum(
        math.comb(discordant, k)
        for k in range(smaller + 1)
    ) / (2.0 ** discordant)

    return min(1.0, 2.0 * cumulative)


def confusion_matrix(
    true_labels: list[str],
    pred_labels: list[str],
) -> np.ndarray:
    index = {
        class_name: position
        for position, class_name in enumerate(CLASS_NAMES)
    }

    matrix = np.zeros(
        (len(CLASS_NAMES), len(CLASS_NAMES)),
        dtype=np.int64,
    )

    for true_class, pred_class in zip(
        true_labels,
        pred_labels,
    ):
        if (
            true_class in index
            and pred_class in index
        ):
            matrix[
                index[true_class],
                index[pred_class],
            ] += 1

    return matrix


def per_class_metrics(
    matrix: np.ndarray,
    method_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        true_positive = int(
            matrix[class_index, class_index]
        )
        false_positive = int(
            matrix[:, class_index].sum()
            - true_positive
        )
        false_negative = int(
            matrix[class_index, :].sum()
            - true_positive
        )
        support = int(matrix[class_index, :].sum())

        precision = (
            true_positive
            / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )
        recall = (
            true_positive
            / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )
        f1 = (
            2.0 * precision * recall
            / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        rows.append(
            {
                "method": method_name,
                "class_name": class_name,
                "support": support,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )

    return rows


def aggregate_metrics(
    matrix: np.ndarray,
    method_name: str,
) -> dict[str, object]:
    rows = per_class_metrics(matrix, method_name)

    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    accuracy = correct / total if total else 0.0
    ci_low, ci_high = wilson_interval(correct, total)

    macro_precision = float(
        np.mean([row["precision"] for row in rows])
    )
    macro_recall = float(
        np.mean([row["recall"] for row in rows])
    )
    macro_f1 = float(
        np.mean([row["f1_score"] for row in rows])
    )

    weighted_f1 = (
        sum(
            float(row["f1_score"]) * int(row["support"])
            for row in rows
        )
        / total
        if total
        else 0.0
    )

    return {
        "method": method_name,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "accuracy_ci95_low": ci_low,
        "accuracy_ci95_high": ci_high,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def save_matrix_csv(
    matrix: np.ndarray,
    path: Path,
) -> None:
    dataframe = pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )
    dataframe.index.name = "true_class"
    dataframe.to_csv(
        path,
        encoding="utf-8-sig",
    )


def plot_confusion_matrix(
    matrix: np.ndarray,
    title: str,
    output_path: Path,
    normalized: bool,
) -> None:
    if normalized:
        row_sums = matrix.sum(
            axis=1,
            keepdims=True,
        )
        plotted = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(
                matrix,
                dtype=np.float64,
            ),
            where=row_sums != 0,
        )
    else:
        plotted = matrix.astype(np.float64)

    figure, axis = plt.subplots(
        figsize=(9.5, 8.0),
    )
    image = axis.imshow(plotted)

    axis.set_title(title)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")

    axis.set_xticks(
        np.arange(len(CLASS_NAMES)),
        labels=CLASS_NAMES,
        rotation=45,
        ha="right",
    )
    axis.set_yticks(
        np.arange(len(CLASS_NAMES)),
        labels=CLASS_NAMES,
    )

    figure.colorbar(image, ax=axis)

    for row_index in range(len(CLASS_NAMES)):
        for column_index in range(len(CLASS_NAMES)):
            if normalized:
                text = f"{plotted[row_index, column_index]:.2f}"
            else:
                text = str(
                    int(matrix[row_index, column_index])
                )

            axis.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
            )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    if not PREDICTIONS_CSV.is_file():
        raise FileNotFoundError(
            f"找不到最终配对结果：{PREDICTIONS_CSV}"
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = pd.read_csv(PREDICTIONS_CSV)

    required_columns = {
        "true_class",
        "yolo_pred_class",
        "classifier_pred_class",
        "yolo_correct",
        "two_stage_correct",
        "status",
    }
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            "paired_predictions.csv缺少字段："
            + ", ".join(sorted(missing_columns))
        )

    processed = data[
        data["status"].astype(str) == "processed"
    ].copy()

    if len(processed) != len(data):
        print(
            f"提示：总记录 {len(data)}，"
            f"成功处理 {len(processed)}。"
        )

    yolo_matrix = confusion_matrix(
        processed["true_class"].astype(str).tolist(),
        processed["yolo_pred_class"].astype(str).tolist(),
    )
    two_stage_matrix = confusion_matrix(
        processed["true_class"].astype(str).tolist(),
        processed[
            "classifier_pred_class"
        ].astype(str).tolist(),
    )

    yolo_metrics = aggregate_metrics(
        yolo_matrix,
        "single_stage_yolo11",
    )
    two_stage_metrics = aggregate_metrics(
        two_stage_matrix,
        "two_stage_yolo11_convnextv2",
    )

    yolo_correct_mask = (
        processed["yolo_correct"]
        .astype(str)
        .str.lower()
        .eq("true")
    )
    two_stage_correct_mask = (
        processed["two_stage_correct"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    both_correct = int(
        (yolo_correct_mask & two_stage_correct_mask).sum()
    )
    both_wrong = int(
        ((~yolo_correct_mask) & (~two_stage_correct_mask)).sum()
    )
    yolo_only_correct = int(
        (yolo_correct_mask & (~two_stage_correct_mask)).sum()
    )
    two_stage_only_correct = int(
        ((~yolo_correct_mask) & two_stage_correct_mask).sum()
    )

    mcnemar_p = exact_mcnemar_p_value(
        yolo_only_correct,
        two_stage_only_correct,
    )

    overall_rows = [
        yolo_metrics,
        two_stage_metrics,
    ]

    overall_csv_path = RESULTS_DIR / "overall_metrics.csv"
    with overall_csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(overall_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(overall_rows)

    per_class_rows = (
        per_class_metrics(
            yolo_matrix,
            "single_stage_yolo11",
        )
        + per_class_metrics(
            two_stage_matrix,
            "two_stage_yolo11_convnextv2",
        )
    )

    per_class_csv_path = (
        RESULTS_DIR
        / "per_class_precision_recall_f1.csv"
    )
    with per_class_csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(per_class_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(per_class_rows)

    save_matrix_csv(
        yolo_matrix,
        RESULTS_DIR / "confusion_matrix_yolo_counts.csv",
    )
    save_matrix_csv(
        two_stage_matrix,
        RESULTS_DIR / "confusion_matrix_two_stage_counts.csv",
    )

    plot_confusion_matrix(
        yolo_matrix,
        "Single-stage YOLO11m confusion matrix",
        RESULTS_DIR / "confusion_matrix_yolo_counts.png",
        normalized=False,
    )
    plot_confusion_matrix(
        yolo_matrix,
        "Single-stage YOLO11m normalized confusion matrix",
        RESULTS_DIR / "confusion_matrix_yolo_normalized.png",
        normalized=True,
    )
    plot_confusion_matrix(
        two_stage_matrix,
        "Two-stage YOLO11m + ConvNeXt V2 confusion matrix",
        RESULTS_DIR / "confusion_matrix_two_stage_counts.png",
        normalized=False,
    )
    plot_confusion_matrix(
        two_stage_matrix,
        "Two-stage YOLO11m + ConvNeXt V2 normalized confusion matrix",
        RESULTS_DIR / "confusion_matrix_two_stage_normalized.png",
        normalized=True,
    )

    summary = {
        "experiment": "final_statistical_analysis",
        "source_predictions": str(PREDICTIONS_CSV),
        "processed_samples": int(len(processed)),
        "single_stage_yolo11": yolo_metrics,
        "two_stage_yolo11_convnextv2": two_stage_metrics,
        "improvement": {
            "correct_images": (
                int(two_stage_metrics["correct"])
                - int(yolo_metrics["correct"])
            ),
            "accuracy_points": (
                float(two_stage_metrics["accuracy"])
                - float(yolo_metrics["accuracy"])
            ),
        },
        "paired_outcomes": {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "yolo_only_correct": yolo_only_correct,
            "two_stage_only_correct": two_stage_only_correct,
        },
        "mcnemar_exact": {
            "discordant_pairs": (
                yolo_only_correct
                + two_stage_only_correct
            ),
            "p_value_two_sided": mcnemar_p,
        },
        "confidence_interval_method": (
            "Wilson score interval, 95%"
        ),
    }

    summary_path = RESULTS_DIR / "statistical_summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n✅ 最终统计分析完成")
    print(
        "方法 | 正确/总数 | 准确率 | 95%置信区间 | "
        "Macro-F1 | Weighted-F1"
    )

    for row in overall_rows:
        print(
            f"{row['method']} | "
            f"{row['correct']}/{row['total']} | "
            f"{row['accuracy'] * 100:.2f}% | "
            f"[{row['accuracy_ci95_low'] * 100:.2f}%, "
            f"{row['accuracy_ci95_high'] * 100:.2f}%] | "
            f"{row['macro_f1'] * 100:.2f}% | "
            f"{row['weighted_f1'] * 100:.2f}%"
        )

    print("\nMcNemar精确检验：")
    print(f"仅YOLO正确：{yolo_only_correct}")
    print(f"仅两阶段正确：{two_stage_only_correct}")
    print(f"双侧p值：{mcnemar_p:.8f}")

    print("\n结果目录：", RESULTS_DIR)
    print("统计汇总：", summary_path)
    print("总体指标：", overall_csv_path)
    print("分类别指标：", per_class_csv_path)


if __name__ == "__main__":
    main()
