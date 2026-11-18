# ============================================================================
# Adaptive Pred-ROI cropping variant.
# Repository name : s05_crop_pred_adaptive.py
# Original name   : crop_pred_adaptive.py
# Paper           : Task-Decoupled Fine-Grained Product Recognition Using
#                   Tight Predicted ROIs for Automated Vending Cabinets
# Note            : algorithm unchanged; comments kept in the original Chinese.
# ============================================================================
import os
import cv2
from ultralytics import YOLO

# 配置
from pathlib import Path
import math
PROJECT_ROOT = Path(__file__).resolve().parent

model_path = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "train_8cls_gpu2"
    / "weights"
    / "best.pt"
)

image_dir = PROJECT_ROOT / "data" / "test"

output_dir = PROJECT_ROOT / "cropped_pred_adaptive_我的GPU版"
conf_threshold = 0.5

def get_adaptive_expand_ratio(
    box_width: float,
    box_height: float,
    image_width: int,
    image_height: int,
) -> float:
    """
    根据检测框相对于整幅图像的大小，自适应确定ROI扩边比例。

    小目标扩边较多，以保留更多上下文；
    大目标扩边较少，以减少无关背景。
    """
    image_area = max(float(image_width * image_height), 1.0)
    box_area = max(float(box_width * box_height), 1.0)

    # 检测框面积占整幅图像面积的比例
    relative_area = min(box_area / image_area, 1.0)

    # 对面积比例开平方，减弱极端大小差异
    scale = math.sqrt(relative_area)

    # 检测框长宽比
    aspect_ratio = max(
        float(box_width) / max(float(box_height), 1.0),
        1e-6,
    )

    # 对过宽或过高的目标稍微增加扩边
    shape_term = min(abs(math.log(aspect_ratio)), 1.0)

    expand_ratio = (
        0.08
        + 0.14 * (1.0 - scale)
        + 0.02 * shape_term
    )

    # 将扩边比例限制在10%～25%
    return max(0.10, min(expand_ratio, 0.25))


model = YOLO(str(model_path))
model = YOLO(str(model_path))

# 清空输出目录，避免旧文件残留
import shutil
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

count = 0
no_detection = 0
processed = set()  # 记录已处理的原始图片名（去重）

for img_name in os.listdir(image_dir):
    if not img_name.endswith('.jpg'):
        continue

    img_path = os.path.join(image_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]
    results = model(img_path, verbose=False)[0]
    boxes = results.boxes

    if boxes is None or len(boxes) == 0:
        no_detection += 1
        continue

    best_box = max(boxes, key=lambda x: x.conf[0])
    conf = float(best_box.conf[0])
    if conf < conf_threshold:
        no_detection += 1
        continue

    cls = int(best_box.cls[0])
    pred_class = model.names[cls]

    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())

    box_w = x2 - x1
    box_h = y2 - y1

    adaptive_ratio = get_adaptive_expand_ratio(
        box_width=box_w,
        box_height=box_h,
        image_width=w,
        image_height=h,
    )

    expand_w = int(box_w * adaptive_ratio)
    expand_h = int(box_h * adaptive_ratio)

    crop_x1 = max(0, x1 - expand_w)
    crop_y1 = max(0, y1 - expand_h)
    crop_x2 = min(w, x2 + expand_w)
    crop_y2 = min(h, y2 + expand_h)

    roi = img[crop_y1:crop_y2, crop_x1:crop_x2]
    if roi.size == 0:
        continue

    # ✅ 用原始文件名（去掉扩展名）保存，保留 (1) 区分
    base_name = os.path.splitext(img_name)[0]
    save_name = f"{base_name}_pred_{pred_class}_conf{conf:.2f}.jpg"
    cv2.imwrite(os.path.join(output_dir, save_name), roi)
    count += 1

print(f"✅ 裁剪完成！")
print(f"   总图片数: {len(os.listdir(image_dir))}")
print(f"   成功裁剪: {count} 张")
print(f"   未检测到/低置信度: {no_detection} 张")
print(f"   保存路径: {output_dir}")