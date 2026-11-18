# ============================================================================
# Locate the YOLO label directory for the test split.
# Repository name : s42_find_test_labels.py
# Original name   : 查找测试集YOLO标签目录.py
# Paper           : Task-Decoupled Fine-Grained Product Recognition Using
#                   Tight Predicted ROIs for Automated Vending Cabinets
# Note            : algorithm unchanged; comments kept in the original Chinese.
# ============================================================================
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path


# 把本脚本放到项目根目录运行
PROJECT_ROOT = Path(__file__).resolve().parent
TEST_IMAGE_DIR = PROJECT_ROOT / "data" / "test"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    if not TEST_IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"找不到测试图像目录：{TEST_IMAGE_DIR}")

    test_stems = {
        path.stem
        for path in TEST_IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }

    print(f"测试图像文件名数量：{len(test_stems)}")
    print("正在整个项目中搜索同名YOLO标签……")

    matches_by_dir: dict[Path, set[str]] = defaultdict(set)

    for txt_path in PROJECT_ROOT.rglob("*.txt"):
        if txt_path.stem in test_stems:
            matches_by_dir[txt_path.parent].add(txt_path.stem)

    if not matches_by_dir:
        print("\n❌ 整个项目中都没有找到与测试图片同名的txt标签。")
        print("说明当前测试集可能没有GT标注，不能直接做GT-ROI实验。")
        return

    ranked = sorted(
        matches_by_dir.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )

    print("\n找到的候选标签目录：")
    for folder, stems in ranked:
        print(f"{len(stems):4d}/{len(test_stems)}  {folder}")

    best_dir, best_stems = ranked[0]
    missing = sorted(test_stems - best_stems)

    print("\n最可能的标签目录：")
    print(best_dir)
    print(f"匹配数量：{len(best_stems)}/{len(test_stems)}")

    if missing:
        print(f"\n仍缺少 {len(missing)} 个标签，前20个文件名：")
        for name in missing[:20]:
            print("  ", name + ".txt")
    else:
        print("\n✅ 680张测试图片均找到对应标签。")
        print("请把GT-ROI脚本中的标签目录改为：")
        print(f'LABEL_DIR = Path(r"{best_dir}")')


if __name__ == "__main__":
    main()
