# ============================================================================
# Train the YOLO11m detector (paper Table II).
# Repository name : s03_train_yolo11m.py
# Original name   : train_yolo_我的GPU版.py
# Paper           : Task-Decoupled Fine-Grained Product Recognition Using
#                   Tight Predicted ROIs for Automated Vending Cabinets
# Note            : algorithm unchanged; comments kept in the original Chinese.
# ============================================================================
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO


# 本文件请放在项目最外层：
# shiyan2-commodity_detect\train_yolo_我的GPU版.py
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
RUNS_ROOT = PROJECT_ROOT / "runs" / "detect"
MODEL_BACKUP_DIR = PROJECT_ROOT / "models" / "yolo11"

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


def find_local_weights() -> Path:
    candidates = [
        PROJECT_ROOT / "yolo11m.pt",
        PROJECT_ROOT / "yolo11n.pt",
        PROJECT_ROOT / "yolo11l.pt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "找不到本地 YOLO 权重。请确认项目根目录存在 yolo11l.pt、yolo11m.pt 或 yolo11n.pt。"
    )


def build_data_yaml() -> Path:
    train_dir = DATA_ROOT / "train_images"
    val_dir = DATA_ROOT / "val_images"
    test_dir = DATA_ROOT / "test"

    for required in (train_dir, val_dir):
        if not required.is_dir():
            raise FileNotFoundError(f"找不到数据目录：{required}")

    yaml_path = PROJECT_ROOT / "data_我的电脑.yaml"
    lines = [
        f"path: {DATA_ROOT.as_posix()}",
        "train: train_images",
        "val: val_images",
    ]
    if test_dir.is_dir():
        lines.append("test: test")

    lines.extend(
        [
            f"nc: {len(CLASS_NAMES)}",
            "names:",
            *[f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)],
        ]
    )
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def main() -> None:
    print("项目目录：", PROJECT_ROOT)
    print("PyTorch：", torch.__version__)
    print("CUDA 可用：", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("当前 Python 环境没有检测到 CUDA，请检查 PyCharm 解释器。")
    print("显卡：", torch.cuda.get_device_name(0))

    weights = find_local_weights()
    data_yaml = build_data_yaml()
    print("预训练权重：", weights)
    print("数据配置：", data_yaml)

    model = YOLO(str(weights))

    # RTX 5060 Laptop 8GB 的保守设置：
    # YOLO11l + 1280 分辨率使用 batch=2；显存不足时改为 batch=1。
    results = model.train(
        data=str(data_yaml),
        epochs=30,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        amp=False,          # 避免 AMP 检查联网下载失败
        cache=False,
        patience=10,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        scale=0.5,
        project=str(RUNS_ROOT),
        name="train_8cls_我的GPU版",
        exist_ok=False,
        save=True,
        plots=True,
        verbose=True,
    )

    run_dir = Path(results.save_dir)
    best_path = run_dir / "weights" / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"训练结束但找不到最佳模型：{best_path}")

    MODEL_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = MODEL_BACKUP_DIR / f"best_yolo11m_{stamp}.pt"
    shutil.copy2(best_path, backup_path)

    print("\n✅ YOLO 训练完成")
    print("训练结果：", run_dir)
    print("原始最佳模型：", best_path)
    print("备份最佳模型：", backup_path)


if __name__ == "__main__":
    main()
