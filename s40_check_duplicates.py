# ============================================================================
# Check train/val/test overlap and near-duplicates.
# Repository name : s40_check_duplicates.py
# Original name   : 检查训练验证测试集重复与近重复.py
# Paper           : Task-Decoupled Fine-Grained Product Recognition Using
#                   Tight Predicted ROIs for Automated Vending Cabinets
# Note            : algorithm unchanged; comments kept in the original Chinese.
# ============================================================================
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
import imagehash


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "results" / "duplicate_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PHASH_DISTANCE_THRESHOLD = 4


@dataclass(frozen=True)
class ImageRecord:
    split: str
    path: Path
    sha256: str
    phash_hex: str
    phash_int: int


class BKNode:
    def __init__(self, value: int, index: int) -> None:
        self.value = value
        self.indexes = [index]
        self.children: dict[int, "BKNode"] = {}


class BKTree:
    def __init__(self) -> None:
        self.root: BKNode | None = None

    @staticmethod
    def distance(a: int, b: int) -> int:
        return (a ^ b).bit_count()

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = BKNode(value, index)
            return

        node = self.root
        while True:
            dist = self.distance(value, node.value)
            if dist == 0:
                node.indexes.append(index)
                return
            if dist not in node.children:
                node.children[dist] = BKNode(value, index)
                return
            node = node.children[dist]

    def query(self, value: int, max_distance: int) -> list[int]:
        if self.root is None:
            return []

        found: list[int] = []
        stack = [self.root]

        while stack:
            node = stack.pop()
            dist = self.distance(value, node.value)

            if dist <= max_distance:
                found.extend(node.indexes)

            lower = dist - max_distance
            upper = dist + max_distance
            for edge_distance, child in node.children.items():
                if lower <= edge_distance <= upper:
                    stack.append(child)

        return found


def resolve_split_dirs() -> dict[str, Path]:
    candidates = {
        "train": [
            DATA_ROOT / "train_images",
            DATA_ROOT / "images" / "train",
            DATA_ROOT / "train",
            PROJECT_ROOT / "train_images",
        ],
        "val": [
            DATA_ROOT / "val_images",
            DATA_ROOT / "images" / "val",
            DATA_ROOT / "valid_images",
            DATA_ROOT / "validation_images",
            DATA_ROOT / "val",
            PROJECT_ROOT / "val_images",
        ],
        "test": [
            DATA_ROOT / "test",
            DATA_ROOT / "test_images",
            DATA_ROOT / "images" / "test",
            PROJECT_ROOT / "test",
        ],
    }

    resolved: dict[str, Path] = {}
    for split, paths in candidates.items():
        for path in paths:
            if path.is_dir():
                resolved[split] = path
                break

    missing = {"train", "val", "test"} - set(resolved)
    if missing:
        searched = "\n".join(
            f"{split}: " + " | ".join(str(p) for p in candidates[split])
            for split in sorted(missing)
        )
        raise FileNotFoundError(
            "未自动找到以下数据目录："
            + ", ".join(sorted(missing))
            + "\n请检查目录名称，脚本搜索过：\n"
            + searched
        )

    return resolved


def iter_images(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def build_records(split_dirs: dict[str, Path]) -> list[ImageRecord]:
    records: list[ImageRecord] = []

    for split, folder in split_dirs.items():
        image_paths = list(iter_images(folder))
        print(f"{split}: 找到 {len(image_paths)} 张图片，正在计算哈希……")

        for index, path in enumerate(image_paths, start=1):
            try:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    phash = imagehash.phash(image, hash_size=8)

                records.append(
                    ImageRecord(
                        split=split,
                        path=path,
                        sha256=sha256_file(path),
                        phash_hex=str(phash),
                        phash_int=int(str(phash), 16),
                    )
                )
            except Exception as exc:
                print(f"警告：无法读取 {path}：{exc}")

            if index % 200 == 0:
                print(f"  已处理 {index}/{len(image_paths)}")

    return records


def write_exact_duplicates(records: list[ImageRecord]) -> tuple[int, int]:
    groups: dict[str, list[ImageRecord]] = {}
    for record in records:
        groups.setdefault(record.sha256, []).append(record)

    duplicate_groups = [group for group in groups.values() if len(group) > 1]

    output_path = OUTPUT_DIR / "exact_duplicates.csv"
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["group_id", "split", "path", "sha256", "cross_split"])

        for group_id, group in enumerate(duplicate_groups, start=1):
            cross_split = len({item.split for item in group}) > 1
            for item in group:
                writer.writerow(
                    [group_id, item.split, str(item.path), item.sha256, cross_split]
                )

    cross_split_groups = sum(
        len({item.split for item in group}) > 1
        for group in duplicate_groups
    )
    return len(duplicate_groups), cross_split_groups


def write_near_duplicates(records: list[ImageRecord]) -> tuple[int, int]:
    tree = BKTree()
    pairs: set[tuple[int, int]] = set()

    for current_index, record in enumerate(records):
        for other_index in tree.query(
            record.phash_int,
            PHASH_DISTANCE_THRESHOLD,
        ):
            left, right = sorted((other_index, current_index))
            if left == right:
                continue
            if records[left].sha256 == records[right].sha256:
                continue
            pairs.add((left, right))

        tree.add(record.phash_int, current_index)

        if (current_index + 1) % 500 == 0:
            print(f"近重复检索：已处理 {current_index + 1}/{len(records)}")

    output_path = OUTPUT_DIR / "near_duplicates.csv"
    cross_split_count = 0

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "pair_id",
                "split_1",
                "path_1",
                "split_2",
                "path_2",
                "phash_distance",
                "cross_split",
                "review",
            ]
        )

        for pair_id, (left, right) in enumerate(sorted(pairs), start=1):
            first = records[left]
            second = records[right]
            distance = BKTree.distance(first.phash_int, second.phash_int)
            cross_split = first.split != second.split
            cross_split_count += int(cross_split)

            writer.writerow(
                [
                    pair_id,
                    first.split,
                    str(first.path),
                    second.split,
                    str(second.path),
                    distance,
                    cross_split,
                    "待人工核对",
                ]
            )

    return len(pairs), cross_split_count


def write_inventory(records: list[ImageRecord]) -> None:
    output_path = OUTPUT_DIR / "image_inventory.csv"
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "path", "sha256", "phash"])
        for item in records:
            writer.writerow([item.split, str(item.path), item.sha256, item.phash_hex])


def main() -> None:
    split_dirs = resolve_split_dirs()

    print("使用的数据目录：")
    for split, path in split_dirs.items():
        print(f"  {split}: {path}")

    records = build_records(split_dirs)
    write_inventory(records)

    exact_groups, exact_cross_groups = write_exact_duplicates(records)
    near_pairs, near_cross_pairs = write_near_duplicates(records)

    split_counts = {
        split: sum(record.split == split for record in records)
        for split in ("train", "val", "test")
    }

    summary = (
        "数据重复与近重复检查结果\n"
        "==========================\n"
        f"train图片数：{split_counts['train']}\n"
        f"val图片数：{split_counts['val']}\n"
        f"test图片数：{split_counts['test']}\n"
        f"完全重复组数：{exact_groups}\n"
        f"跨集合完全重复组数：{exact_cross_groups}\n"
        f"pHash距离≤{PHASH_DISTANCE_THRESHOLD}的近重复对数：{near_pairs}\n"
        f"其中跨集合近重复对数：{near_cross_pairs}\n\n"
        "说明：near_duplicates.csv中的近重复需要人工查看，"
        "不能仅凭哈希距离自动删除。\n"
    )

    summary_path = OUTPUT_DIR / "duplicate_summary.txt"
    summary_path.write_text(summary, encoding="utf-8-sig")

    print("\n" + summary)
    print("结果目录：", OUTPUT_DIR)


if __name__ == "__main__":
    main()
