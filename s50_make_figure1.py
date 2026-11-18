# ============================================================================
# Generate the framework / sample figure (paper Fig. 1-2).
# Repository name : s50_make_figure1.py
# Original name   : make_figure1_sample_auto.py
# Paper           : Task-Decoupled Fine-Grained Product Recognition Using
#                   Tight Predicted ROIs for Automated Vending Cabinets
# Note            : algorithm unchanged; comments kept in the original Chinese.
# ============================================================================
# -*- coding: utf-8 -*-
"""
自动生成论文图1所需的两张真实图片：
1) 02_yolo_detection.jpg  —— 带最高置信度检测框的场景图
2) 03_tight_pred_roi.jpg   —— ρ=0 的 Tight Pred-ROI

使用方法：
- 把本文件放在 PyCharm 项目根目录直接运行。
- 程序会自动搜索 YOLO 检测器 best.pt 和 hks_large493.jpg。
- 如果没找到，会自动弹出文件选择窗口，不需要手动改路径。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import sys

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
TARGET_IMAGE_NAME = "hks_large493"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figure1_sample"

CONF_THRESHOLD = 0.01
IMAGE_SIZE = 640


def choose_file(title: str, filetypes: list[tuple[str, str]]) -> Path:
    """弹出 Windows 文件选择窗口。"""
    try:
        from tkinter import Tk, filedialog
    except ImportError as exc:
        raise RuntimeError("当前 Python 环境缺少 tkinter，无法弹出文件选择窗口。") from exc

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilename(
        title=title,
        initialdir=str(PROJECT_ROOT),
        filetypes=filetypes,
    )
    root.destroy()

    if not selected:
        raise RuntimeError(f"未选择文件：{title}")
    return Path(selected)


def find_yolo_weight(root: Path) -> Path:
    """
    自动寻找 YOLO 检测器权重。
    优先级：
    1. runs/detect/**/weights/best.pt
    2. 任意目录中的 best.pt
    3. 弹窗手动选择 .pt
    """
    candidates: list[Path] = []

    for p in root.rglob("best.pt"):
        if not p.is_file():
            continue
        lower_parts = [part.lower() for part in p.parts]
        score = 0
        if "runs" in lower_parts:
            score += 5
        if "detect" in lower_parts:
            score += 8
        if "weights" in lower_parts:
            score += 10
        if p.name.lower() == "best.pt":
            score += 3
        candidates.append((score, p))

    if candidates:
        candidates.sort(key=lambda item: (-item[0], len(str(item[1]))))
        best_score, best_path = candidates[0]
        print(f"[自动找到检测器权重] {best_path}")
        return best_path

    print("[未自动找到 best.pt] 请在弹窗中选择 YOLO 检测器权重。")
    return choose_file(
        "请选择 YOLO 检测器权重 best.pt",
        [("YOLO 权重", "*.pt"), ("所有文件", "*.*")],
    )


def find_target_image(root: Path, stem: str) -> Path:
    """
    自动寻找目标图片。
    支持 jpg/jpeg/png/bmp/webp。
    找不到时弹窗选择。
    """
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    matches = [
        p for p in root.rglob("*")
        if p.is_file() and p.stem.lower() == stem.lower() and p.suffix.lower() in extensions
    ]

    if matches:
        # 优先选择项目根目录或“论文图片”目录中的图片
        def rank(p: Path) -> tuple[int, int]:
            text = str(p).lower()
            score = 0
            if "论文图片" in text:
                score += 5
            if p.parent == root:
                score += 4
            return (-score, len(str(p)))

        matches.sort(key=rank)
        image_path = matches[0]
        print(f"[自动找到原始图片] {image_path}")
        return image_path

    print(f"[未自动找到 {stem}] 请在弹窗中选择原始场景图。")
    return choose_file(
        f"请选择原始场景图（建议选择 {stem}.jpg）",
        [
            ("图像文件", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("所有文件", "*.*"),
        ],
    )


def imread_unicode(path: Path) -> np.ndarray:
    """兼容中文路径的 OpenCV 读图。"""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV 无法读取图像：{path}")
    return image


def imwrite_unicode(path: Path, image: np.ndarray, quality: int = 95) -> None:
    """兼容中文路径的 OpenCV 保存。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        ok, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
    elif suffix == ".png":
        ok, encoded = cv2.imencode(
            ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3]
        )
    else:
        ok, encoded = cv2.imencode(suffix, image)

    if not ok:
        raise RuntimeError(f"图像编码失败：{path}")
    encoded.tofile(str(path))


def draw_best_detection(
    image: np.ndarray,
    box_xyxy: tuple[int, int, int, int],
    class_name: str,
    confidence: float,
) -> np.ndarray:
    """只绘制最高置信度预测框。"""
    x1, y1, x2, y2 = box_xyxy
    output = image.copy()

    # 论文图中采用清晰的蓝色框，避免过于花哨
    box_color = (180, 85, 20)  # BGR
    text_bg = (180, 85, 20)
    text_color = (255, 255, 255)

    h, w = output.shape[:2]
    line_width = max(2, round(min(h, w) / 250))
    cv2.rectangle(output, (x1, y1), (x2, y2), box_color, line_width)

    label = f"{class_name}  {confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, min(h, w) / 850)
    thickness = max(1, line_width - 1)

    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    pad = 5
    label_y1 = max(0, y1 - th - baseline - 2 * pad)
    label_x2 = min(w - 1, x1 + tw + 2 * pad)

    cv2.rectangle(
        output,
        (x1, label_y1),
        (label_x2, y1),
        text_bg,
        -1,
    )
    cv2.putText(
        output,
        label,
        (x1 + pad, y1 - baseline - pad),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )
    return output


def main() -> None:
    print("=" * 68)
    print("开始生成论文图1真实图像素材")
    print(f"项目目录：{PROJECT_ROOT}")
    print("=" * 68)

    model_path = find_yolo_weight(PROJECT_ROOT)
    image_path = find_target_image(PROJECT_ROOT, TARGET_IMAGE_NAME)

    if model_path.suffix.lower() != ".pt":
        raise RuntimeError(
            f"选择的文件不是 YOLO .pt 权重：{model_path}\n"
            "请不要选择 best_classifier_new.pth 或其他分类器权重。"
        )

    original = imread_unicode(image_path)
    height, width = original.shape[:2]

    print("\n[加载模型并推理]")
    print(f"模型：{model_path}")
    print(f"图片：{image_path}")

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        imgsz=IMAGE_SIZE,
        verbose=False,
    )

    if not results:
        raise RuntimeError("模型未返回预测结果。")

    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        raise RuntimeError(
            f"在 conf={CONF_THRESHOLD} 下未检测到目标。\n"
            "请确认：\n"
            "1. 选择的是 YOLO 检测器 best.pt；\n"
            "2. 图片确实属于该检测数据集。"
        )

    best_idx = int(boxes.conf.argmax().item())
    xyxy = boxes.xyxy[best_idx].detach().cpu().numpy().round().astype(int)
    x1, y1, x2, y2 = map(int, xyxy.tolist())

    # 边界约束
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    confidence = float(boxes.conf[best_idx].item())
    class_id = int(boxes.cls[best_idx].item())
    class_name = str(result.names[class_id])

    annotated = draw_best_detection(
        original,
        (x1, y1, x2, y2),
        class_name,
        confidence,
    )

    tight_roi = original[y1:y2, x1:x2].copy()
    if tight_roi.size == 0:
        raise RuntimeError("紧致 ROI 为空，请检查预测框。")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 保存原图副本，方便后续直接做图1
    original_out = OUTPUT_DIR / "01_scene_image.jpg"
    detection_out = OUTPUT_DIR / "02_yolo_detection.jpg"
    roi_out = OUTPUT_DIR / "03_tight_pred_roi.jpg"

    imwrite_unicode(original_out, original)
    imwrite_unicode(detection_out, annotated)
    imwrite_unicode(roi_out, tight_roi)

    # 保存预测信息
    info_out = OUTPUT_DIR / "prediction_info.txt"
    info_out.write_text(
        "\n".join(
            [
                f"model={model_path}",
                f"image={image_path}",
                f"class_id={class_id}",
                f"class_name={class_name}",
                f"confidence={confidence:.6f}",
                f"box_xyxy={x1},{y1},{x2},{y2}",
                f"threshold={CONF_THRESHOLD}",
                "roi_expansion=0",
            ]
        ),
        encoding="utf-8",
    )

    print("\n生成完成：")
    print(f"1. 原始场景图：{original_out}")
    print(f"2. YOLO 检测图：{detection_out}")
    print(f"3. Tight Pred-ROI：{roi_out}")
    print(f"4. 预测信息：{info_out}")
    print()
    print(f"预测类别：{class_name}")
    print(f"置信度：{confidence:.4f}")
    print(f"预测框：({x1}, {y1}, {x2}, {y2})")
    print("=" * 68)

    # Windows 自动打开结果文件夹
    try:
        import os
        os.startfile(str(OUTPUT_DIR))
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[运行失败]")
        print(exc)
        print("\n按 Enter 键退出……")
        input()
        sys.exit(1)
