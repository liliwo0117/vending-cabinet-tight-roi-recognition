# ============================================================================
# Export suspected near-duplicate pairs for manual review.
# Repository name : s41_export_near_duplicates.py
# Original name   : 导出跨集合近重复样本供人工核对.py
# Paper           : Task-Decoupled Fine-Grained Product Recognition Using
#                   Tight Predicted ROIs for Automated Vending Cabinets
# Note            : algorithm unchanged; comments kept in the original Chinese.
# ============================================================================
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


# 把本脚本放在项目根目录：
PROJECT_ROOT = Path(__file__).resolve().parent

CSV_PATH = PROJECT_ROOT / "results" / "duplicate_check" / "near_duplicates.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "duplicate_check" / "cross_split_review"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MONTAGE_PATH = OUTPUT_DIR / "跨集合近重复人工核对图.png"


# --- portability: resolve a CJK font across platforms (added during packaging) ---
def _resolve_font_path():
    import os as _os
    for _p in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ):
        if _os.path.exists(_p):
            return _p
    return None


FONT_PATH = _resolve_font_path()
# ---------------------------------------------------------------------------


def get_font(size: int) -> ImageFont.ImageFont:
    candidates = [Path(p) for p in (FONT_PATH,) if p]
    candidates.append(Path(r"C:\Windows\Fonts\simsun.ttc"))
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def copy_image(source: Path, pair_id: int, side: str, split: str) -> Path:
    destination = OUTPUT_DIR / (
        f"pair_{pair_id}_{side}_{split}_{source.name}"
    )
    shutil.copy2(source, destination)
    return destination


def make_montage(pairs: list[tuple[int, Path, Path, int]]) -> None:
    if not pairs:
        print("没有跨集合近重复样本，不生成组合图。")
        return

    tile_w = 720
    image_h = 500
    caption_h = 105
    gap = 30
    margin = 30

    rows = len(pairs)
    canvas_w = margin * 2 + tile_w * 2 + gap
    canvas_h = margin * 2 + rows * (image_h + caption_h) + (rows - 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = get_font(28)
    text_font = get_font(22)

    for row_index, (pair_id, left_path, right_path, distance) in enumerate(pairs):
        y = margin + row_index * (image_h + caption_h + gap)

        for col_index, image_path in enumerate((left_path, right_path)):
            x = margin + col_index * (tile_w + gap)
            image = Image.open(image_path).convert("RGB")
            fitted = ImageOps.contain(
                image,
                (tile_w - 20, image_h - 20),
                method=Image.Resampling.LANCZOS,
            )

            panel = Image.new("RGB", (tile_w, image_h), "white")
            px = (tile_w - fitted.width) // 2
            py = (image_h - fitted.height) // 2
            panel.paste(fitted, (px, py))
            canvas.paste(panel, (x, y))
            draw.rectangle(
                [x, y, x + tile_w - 1, y + image_h - 1],
                outline="black",
                width=2,
            )

            split = "train" if col_index == 0 else "test"
            caption = (
                f"Pair {pair_id} | {split}\n"
                f"{image_path.name}"
            )
            draw.multiline_text(
                (x + 10, y + image_h + 10),
                caption,
                fill="black",
                font=text_font,
                spacing=6,
            )

        draw.text(
            (margin, y - 2),
            f"pHash distance = {distance}",
            fill="black",
            font=title_font,
        )

    canvas.save(MONTAGE_PATH, dpi=(300, 300))
    print("已生成核对图：", MONTAGE_PATH)


def main() -> None:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"找不到：{CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    cross_df = df[df["cross_split"].astype(str).str.lower() == "true"].copy()

    if cross_df.empty:
        print("没有跨集合近重复样本。")
        return

    montage_pairs: list[tuple[int, Path, Path, int]] = []

    for _, row in cross_df.iterrows():
        pair_id = int(row["pair_id"])
        left_source = Path(str(row["path_1"]))
        right_source = Path(str(row["path_2"]))
        distance = int(row["phash_distance"])

        if not left_source.is_file():
            raise FileNotFoundError(f"找不到：{left_source}")
        if not right_source.is_file():
            raise FileNotFoundError(f"找不到：{right_source}")

        left_copy = copy_image(
            left_source,
            pair_id,
            "A",
            str(row["split_1"]),
        )
        right_copy = copy_image(
            right_source,
            pair_id,
            "B",
            str(row["split_2"]),
        )
        montage_pairs.append((pair_id, left_copy, right_copy, distance))

        print(
            f"Pair {pair_id}: {left_source.name} <-> "
            f"{right_source.name}, distance={distance}"
        )

    make_montage(montage_pairs)
    print("人工核对目录：", OUTPUT_DIR)


if __name__ == "__main__":
    main()
