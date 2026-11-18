# ============================================================================
# Parameters, FLOPs, latency and FPS (paper Sec. IV-E).
# Repository name : s34_efficiency_profile.py
# Original name   : 统计模型参数_FLOPs_推理时间.py
# Paper           : Task-Decoupled Fine-Grained Product Recognition Using
#                   Tight Predicted ROIs for Automated Vending Cabinets
# Note            : algorithm unchanged; comments kept in the original Chinese.
# ============================================================================
from __future__ import annotations

import copy
import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import cv2
import timm
import torch
from PIL import Image
from torchvision import datasets, transforms
from ultralytics import YOLO


# ============================================================
# 将本脚本放在项目根目录运行。
#(repository root)
# 统计内容：
# 1. YOLO11m、ConvNeXt V2-Atto及组合模型的参数量；
# 2. MACs与按2 FLOPs/MAC换算的近似GFLOPs；
# 3. batch=1、FP32、排除磁盘读取的GPU时延；
# 4. 单阶段YOLO、分类器、完整两阶段流水线的平均/中位/P95时延及FPS。
#(repository root)
# 最终参数已经由验证集锁定：
# 检测阈值 = 0.01
# ROI扩展比例 = 0%
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

FINAL_ROI_DIR = (
    PROJECT_ROOT
    / "results"
    / "final_test_conf_0_01_tight_roi"
    / "saved_tight_rois"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "efficiency_profile_final"
)

CONF_THRESHOLD = 0.01
ROI_EXPANSION = 0.0
DETECTOR_INPUT_SIZE = 640
CLASSIFIER_INPUT_SIZE = 224

WARMUP_RUNS = 30
MAX_BENCHMARK_IMAGES = 680
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_state_dict_compat(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total": int(total),
        "trainable": int(trainable),
    }


def profile_macs(
    model: torch.nn.Module,
    input_shape: tuple[int, int, int, int],
) -> dict[str, object]:
    try:
        from thop import profile
    except ImportError:
        return {
            "macs": None,
            "gmacs": None,
            "approx_flops": None,
            "approx_gflops": None,
            "error": (
                "未安装thop。请在当前Python环境执行："
                " pip install thop"
            ),
        }

    try:
        profile_model = copy.deepcopy(model).cpu().float().eval()
        dummy = torch.zeros(input_shape, dtype=torch.float32)

        with torch.inference_mode():
            macs, _ = profile(
                profile_model,
                inputs=(dummy,),
                verbose=False,
            )

        del profile_model
        del dummy

        macs = float(macs)
        approx_flops = 2.0 * macs

        return {
            "macs": macs,
            "gmacs": macs / 1e9,
            "approx_flops": approx_flops,
            "approx_gflops": approx_flops / 1e9,
            "error": "",
        }
    except Exception as exc:
        return {
            "macs": None,
            "gmacs": None,
            "approx_flops": None,
            "approx_gflops": None,
            "error": str(exc),
        }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = int(round((len(sorted_values) - 1) * q))
    return float(sorted_values[index])


def summarize_times(times_ms: list[float]) -> dict[str, float]:
    if not times_ms:
        return {
            "samples": 0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "fps_from_mean": 0.0,
        }

    mean_ms = statistics.mean(times_ms)

    return {
        "samples": len(times_ms),
        "mean_ms": float(mean_ms),
        "median_ms": float(statistics.median(times_ms)),
        "p95_ms": percentile(times_ms, 0.95),
        "fps_from_mean": (
            float(1000.0 / mean_ms)
            if mean_ms > 0
            else 0.0
        ),
    }


def classifier_forward(
    roi_bgr,
    classifier: torch.nn.Module,
    transform,
    device: torch.device,
) -> tuple[int, float]:
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(roi_rgb)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = classifier(tensor)
        probabilities = torch.softmax(logits, dim=1)
        pred_index = int(probabilities.argmax(dim=1).item())
        confidence = float(probabilities[0, pred_index].item())

    return pred_index, confidence


def get_best_box(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    return max(
        boxes,
        key=lambda box: box.conf[0],
    )


def benchmark_detector(
    detector: YOLO,
    image_paths: list[Path],
    device: torch.device,
    yolo_device: int | str,
) -> tuple[dict[str, float], int]:
    print("\n开始统计单阶段YOLO时延……")

    warmup_paths = image_paths[: min(WARMUP_RUNS, len(image_paths))]
    for path in warmup_paths:
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            continue

        detector.predict(
            source=image_bgr,
            conf=CONF_THRESHOLD,
            device=yolo_device,
            save=False,
            verbose=False,
        )
    synchronize(device)

    times_ms: list[float] = []
    failed = 0

    for index, path in enumerate(image_paths, start=1):
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            failed += 1
            continue

        synchronize(device)
        start = time.perf_counter()

        try:
            result = detector.predict(
                source=image_bgr,
                conf=CONF_THRESHOLD,
                device=yolo_device,
                save=False,
                verbose=False,
            )[0]
            _ = get_best_box(result)
        except Exception:
            failed += 1
            continue

        synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)

        if index % 100 == 0:
            print(f"YOLO时延：{index}/{len(image_paths)}")

    return summarize_times(times_ms), failed


def benchmark_classifier(
    classifier: torch.nn.Module,
    transform,
    roi_paths: list[Path],
    device: torch.device,
) -> tuple[dict[str, float], int]:
    print("\n开始统计ConvNeXt V2分类时延……")

    warmup_paths = roi_paths[: min(WARMUP_RUNS, len(roi_paths))]
    for path in warmup_paths:
        roi_bgr = cv2.imread(str(path))
        if roi_bgr is None:
            continue

        classifier_forward(
            roi_bgr,
            classifier,
            transform,
            device,
        )
    synchronize(device)

    times_ms: list[float] = []
    failed = 0

    for index, path in enumerate(roi_paths, start=1):
        # 磁盘读取放在计时之外。
        roi_bgr = cv2.imread(str(path))
        if roi_bgr is None:
            failed += 1
            continue

        synchronize(device)
        start = time.perf_counter()

        try:
            classifier_forward(
                roi_bgr,
                classifier,
                transform,
                device,
            )
        except Exception:
            failed += 1
            continue

        synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)

        if index % 100 == 0:
            print(
                f"分类器时延：{index}/{len(roi_paths)}"
            )

    return summarize_times(times_ms), failed


def benchmark_two_stage(
    detector: YOLO,
    classifier: torch.nn.Module,
    transform,
    image_paths: list[Path],
    device: torch.device,
    yolo_device: int | str,
) -> tuple[dict[str, float], dict[str, int]]:
    print("\n开始统计完整两阶段流水线时延……")

    def run_one(image_bgr) -> str:
        result = detector.predict(
            source=image_bgr,
            conf=CONF_THRESHOLD,
            device=yolo_device,
            save=False,
            verbose=False,
        )[0]

        best_box = get_best_box(result)
        if best_box is None:
            return "missed"

        height, width = image_bgr.shape[:2]
        x1, y1, x2, y2 = map(
            int,
            best_box.xyxy[0].tolist(),
        )

        # 验证集锁定为0%扩展，即紧致预测框。
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        roi = image_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return "empty_roi"

        classifier_forward(
            roi,
            classifier,
            transform,
            device,
        )
        return "ok"

    warmup_paths = image_paths[: min(WARMUP_RUNS, len(image_paths))]
    for path in warmup_paths:
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            continue
        try:
            run_one(image_bgr)
        except Exception:
            pass
    synchronize(device)

    times_ms: list[float] = []
    status_counts = {
        "ok": 0,
        "missed": 0,
        "empty_roi": 0,
        "failed": 0,
        "image_read_failed": 0,
    }

    for index, path in enumerate(image_paths, start=1):
        # 磁盘读取放在计时之外。
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            status_counts["image_read_failed"] += 1
            continue

        synchronize(device)
        start = time.perf_counter()

        try:
            status = run_one(image_bgr)
        except Exception:
            status_counts["failed"] += 1
            continue

        synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)
        status_counts[status] += 1

        if index % 100 == 0:
            print(
                f"两阶段时延：{index}/{len(image_paths)}"
            )

    return summarize_times(times_ms), status_counts


def main() -> None:
    print("YOLO权重：", YOLO_MODEL_PATH)
    print("分类器权重：", CLASSIFIER_MODEL_PATH)
    print("测试集目录：", TEST_IMAGE_DIR)
    print("最终ROI目录：", FINAL_ROI_DIR)
    print("结果目录：", RESULTS_DIR)
    print("检测阈值：", CONF_THRESHOLD)
    print("ROI扩展比例：0%")
    print("batch size：1")
    print("数值精度：FP32")
    print("磁盘读取：不计入推理时延")

    for path, description in [
        (YOLO_MODEL_PATH, "YOLO权重"),
        (CLASSIFIER_MODEL_PATH, "分类器权重"),
        (TRAIN_DATA_DIR, "分类训练目录"),
        (TEST_IMAGE_DIR, "测试集目录"),
        (FINAL_ROI_DIR, "最终测试ROI目录"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"找不到{description}：{path}"
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    yolo_device: int | str = (
        0 if torch.cuda.is_available() else "cpu"
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print("运行设备：", device)
    if device.type == "cuda":
        print("显卡：", torch.cuda.get_device_name(0))

    detector = YOLO(str(YOLO_MODEL_PATH))

    dataset = datasets.ImageFolder(str(TRAIN_DATA_DIR))
    class_names = list(dataset.classes)

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
            transforms.Resize(
                (
                    CLASSIFIER_INPUT_SIZE,
                    CLASSIFIER_INPUT_SIZE,
                )
            ),
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
    )[:MAX_BENCHMARK_IMAGES]

    roi_paths = sorted(
        path
        for path in FINAL_ROI_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )[:MAX_BENCHMARK_IMAGES]

    print("用于时延统计的测试图数量：", len(image_paths))
    print("用于分类时延统计的ROI数量：", len(roi_paths))

    # ---------------- 参数量与计算量 ----------------
    detector_parameters = count_parameters(detector.model)
    classifier_parameters = count_parameters(classifier)

    print("\n开始统计MACs/FLOPs……")
    detector_compute = profile_macs(
        detector.model,
        (
            1,
            3,
            DETECTOR_INPUT_SIZE,
            DETECTOR_INPUT_SIZE,
        ),
    )
    classifier_compute = profile_macs(
        classifier,
        (
            1,
            3,
            CLASSIFIER_INPUT_SIZE,
            CLASSIFIER_INPUT_SIZE,
        ),
    )

    combined_parameters = {
        "total": (
            detector_parameters["total"]
            + classifier_parameters["total"]
        ),
        "trainable": (
            detector_parameters["trainable"]
            + classifier_parameters["trainable"]
        ),
    }

    if (
        detector_compute["macs"] is not None
        and classifier_compute["macs"] is not None
    ):
        combined_macs = (
            float(detector_compute["macs"])
            + float(classifier_compute["macs"])
        )
        combined_compute = {
            "macs": combined_macs,
            "gmacs": combined_macs / 1e9,
            "approx_flops": 2.0 * combined_macs,
            "approx_gflops": 2.0 * combined_macs / 1e9,
            "error": "",
        }
    else:
        combined_compute = {
            "macs": None,
            "gmacs": None,
            "approx_flops": None,
            "approx_gflops": None,
            "error": "一个或多个子模型的MACs统计失败。",
        }

    # ---------------- 推理时延 ----------------
    detector_latency, detector_failed = benchmark_detector(
        detector,
        image_paths,
        device,
        yolo_device,
    )

    classifier_latency, classifier_failed = benchmark_classifier(
        classifier,
        transform,
        roi_paths,
        device,
    )

    two_stage_latency, two_stage_status = benchmark_two_stage(
        detector,
        classifier,
        transform,
        image_paths,
        device,
        yolo_device,
    )

    environment = {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "timm": timm.__version__,
        "opencv": cv2.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
        "cudnn": (
            torch.backends.cudnn.version()
            if torch.cuda.is_available()
            else None
        ),
    }

    summary = {
        "experiment": "model_complexity_and_latency_final",
        "configuration": {
            "confidence_threshold": CONF_THRESHOLD,
            "roi_expansion_ratio": ROI_EXPANSION,
            "batch_size": 1,
            "precision": "FP32",
            "detector_input_size": DETECTOR_INPUT_SIZE,
            "classifier_input_size": CLASSIFIER_INPUT_SIZE,
            "warmup_runs": WARMUP_RUNS,
            "benchmark_images": len(image_paths),
            "disk_io_included": False,
            "flops_convention": (
                "THOP reports MACs; approximate FLOPs = 2 × MACs."
            ),
        },
        "environment": environment,
        "models": {
            "yolo11m_detector": {
                "parameters": detector_parameters,
                "compute": detector_compute,
                "latency": detector_latency,
                "failed_samples": detector_failed,
                "model_path": str(YOLO_MODEL_PATH),
            },
            "convnextv2_atto_classifier": {
                "parameters": classifier_parameters,
                "compute": classifier_compute,
                "latency": classifier_latency,
                "failed_samples": classifier_failed,
                "model_path": str(CLASSIFIER_MODEL_PATH),
            },
            "two_stage_combined": {
                "parameters": combined_parameters,
                "compute": combined_compute,
                "latency": two_stage_latency,
                "status_counts": two_stage_status,
            },
        },
    }

    summary_path = RESULTS_DIR / "efficiency_summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    table_rows = []
    for model_name, item in summary["models"].items():
        parameters = item["parameters"]
        compute = item["compute"]
        latency = item["latency"]

        table_rows.append(
            {
                "model": model_name,
                "parameters": parameters["total"],
                "parameters_million": (
                    parameters["total"] / 1e6
                ),
                "gmacs": compute.get("gmacs"),
                "approx_gflops": compute.get(
                    "approx_gflops"
                ),
                "mean_latency_ms": latency["mean_ms"],
                "median_latency_ms": latency["median_ms"],
                "p95_latency_ms": latency["p95_ms"],
                "fps": latency["fps_from_mean"],
                "latency_samples": latency["samples"],
            }
        )

    csv_path = RESULTS_DIR / "efficiency_table.csv"
    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(table_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(table_rows)

    print("\n✅ 模型复杂度与时延统计完成")
    print(
        "模型 | 参数量(M) | GMACs | 近似GFLOPs | "
        "平均时延(ms) | 中位时延(ms) | P95(ms) | FPS"
    )

    for row in table_rows:
        gmacs_text = (
            f"{row['gmacs']:.3f}"
            if row["gmacs"] is not None
            else "统计失败"
        )
        gflops_text = (
            f"{row['approx_gflops']:.3f}"
            if row["approx_gflops"] is not None
            else "统计失败"
        )

        print(
            f"{row['model']} | "
            f"{row['parameters_million']:.3f} | "
            f"{gmacs_text} | "
            f"{gflops_text} | "
            f"{row['mean_latency_ms']:.3f} | "
            f"{row['median_latency_ms']:.3f} | "
            f"{row['p95_latency_ms']:.3f} | "
            f"{row['fps']:.2f}"
        )

    if detector_compute["error"]:
        print(
            "\nYOLO MACs统计提示：",
            detector_compute["error"],
        )
    if classifier_compute["error"]:
        print(
            "\n分类器MACs统计提示：",
            classifier_compute["error"],
        )

    print("\n硬件：", environment["gpu"])
    print("结果目录：", RESULTS_DIR)
    print("完整汇总：", summary_path)
    print("论文表格：", csv_path)


if __name__ == "__main__":
    main()
