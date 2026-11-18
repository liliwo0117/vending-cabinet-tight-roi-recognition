# ============================================================================
# Assemble cls_data_new/train,val from GT crops.
# Repository name : s02_split_cls_data.py
# Original name   : split_cls_data.py
# Paper           : Task-Decoupled Fine-Grained Product Recognition Using
#                   Tight Predicted ROIs for Automated Vending Cabinets
# Note            : algorithm unchanged; comments kept in the original Chinese.
# ============================================================================
import os
import shutil

# 源数据路径
train_src = r"cropped_gt_train_new"
val_src = r"cropped_gt_val_pred"

# 目标路径
dst_dir = r"cls_data_new"

# 删除旧数据
if os.path.exists(dst_dir):
    shutil.rmtree(dst_dir)

# 创建目录
os.makedirs(os.path.join(dst_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(dst_dir, 'val'), exist_ok=True)

# 复制训练集
print("正在复制训练集...")
for cls in os.listdir(train_src):
    src_cls_path = os.path.join(train_src, cls)
    if not os.path.isdir(src_cls_path):
        continue

    dst_cls_path = os.path.join(dst_dir, 'train', cls)
    os.makedirs(dst_cls_path, exist_ok=True)

    for img in os.listdir(src_cls_path):
        if img.endswith('.jpg'):
            src = os.path.join(src_cls_path, img)
            dst = os.path.join(dst_cls_path, img)
            shutil.copy2(src, dst)

    print(f"  {cls}: {len(os.listdir(dst_cls_path))} 张")

# 复制验证集
print("正在复制验证集...")
for cls in os.listdir(val_src):
    src_cls_path = os.path.join(val_src, cls)
    if not os.path.isdir(src_cls_path):
        continue

    dst_cls_path = os.path.join(dst_dir, 'val', cls)
    os.makedirs(dst_cls_path, exist_ok=True)

    for img in os.listdir(src_cls_path):
        if img.endswith('.jpg'):
            src = os.path.join(src_cls_path, img)
            dst = os.path.join(dst_cls_path, img)
            shutil.copy2(src, dst)

    print(f"  {cls}: {len(os.listdir(dst_cls_path))} 张")

print(f"\n✅ 分类数据集整理完成！保存在 {dst_dir}")