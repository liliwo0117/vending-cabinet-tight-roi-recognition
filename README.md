# Task-Decoupled Fine-Grained Product Recognition Using Tight Predicted ROIs

Reference implementation for

> **Task-Decoupled Fine-Grained Product Recognition Using Tight Predicted ROIs for
> Automated Vending Cabinets**
> Yihui Li, Shaojun Tang, Zihan Shen, Yuhang Ge, and Kai Xie
> School of Electronic Information and Electrical Engineering, Yangtze University

This repository contains the complete experiment code, the fixed configuration, and the
result files behind every number reported in the paper. The code is released so that the
reported evaluation can be re-run and audited, as required for peer review.

---

## 1. What the method does

The pipeline splits fine-grained product recognition into three decoupled modules:

| Module | Role | Implementation |
|---|---|---|
| **A** | product localization | YOLO11m, highest-confidence box only |
| **B** | tight Pred-ROI construction | crop the predicted box, expansion ratio `rho` |
| **C** | fine-grained classification | ConvNeXt V2-Atto (official ImageNet init) |

The detection confidence threshold `tau` and the ROI expansion ratio `rho` are selected
**only on the validation split** and then frozen (`tau* = 0.01`, `rho* = 0`). The independent
test split is touched exactly once. The single-stage YOLO11m baseline and the proposed
cascade reuse **the same detections**, so the comparison isolates the effect of the
reclassification stage alone.

## 2. Headline results (independent test split, 680 images, 8 classes)

| Metric | Single-stage YOLO11m | YOLO11m + tight Pred-ROI + ConvNeXt V2 | Delta |
|---|---|---|---|
| Detection coverage `R_det` | 100.00 % | 100.00 % | – |
| Complete end-to-end accuracy `A_e2e` | 88.53 % (602/680) | **92.94 % (632/680)** | +4.41 pp |
| Macro-F1 | 87.91 % | 92.93 % | +5.02 pp |
| 95 % Wilson CI | 85.91 – 90.71 % | 90.77 – 94.63 % | – |
| Paired McNemar (two-sided) | – | p = 0.000176 | – |
| Parameters | 20.06 M | 23.45 M | +3.39 M |
| Latency / throughput (bs=1, FP32) | 17.79 ms | 26.00 ms / 38.46 FPS | – |

Discordant pairs: 16 images correct only for the baseline, 46 only for the cascade.

Every figure above is reproducible from the JSON/CSV files shipped under [`results/`](results)
— see [`docs/SCRIPT_MAP.md`](docs/SCRIPT_MAP.md) for the exact file that backs each table row
and each figure panel.

## 3. Repository layout

```
.
├── s00 - s02   data preparation (GT-ROI crops, classification dataset assembly)
├── s03 - s05   Module A training + Module B Pred-ROI cropping
├── s06 - s08   Module C (ConvNeXt V2-Atto) training
├── s10 - s13   end-to-end evaluation variants
├── s20 - s25   ablations (confidence threshold, ROI expansion, joint, boundary)
├── s30 - s34   final paired test, statistics, efficiency profiling   <-- paper results
├── s40 - s42   dataset hygiene (exact/near-duplicate checks)
├── s50 - s56   figures and diagnostic utilities
├── config/     dataset description (data.yaml)
├── figures/    the nine figures embedded in the manuscript
├── results/    result files backing the paper (JSON / CSV only, ~3 MB)
├── runs/       detector training arguments and curve data
├── data/       not shipped - see data/README.md
└── weights/    not shipped - see weights/README.md
```

Original script names were in Chinese; each file carries a header stating its original name,
and [`docs/SCRIPT_MAP.md`](docs/SCRIPT_MAP.md) contains the full mapping. Comments inside the
scripts are kept in the original Chinese so that the shipped code stays byte-comparable with
the version that produced the results.

## 4. Environment

Measured on the machine that produced the paper:

| Item | Version |
|---|---|
| OS | Windows 11 (10.0.26100) |
| Python | 3.10.5 |
| PyTorch | 2.11.0 + cu128 |
| timm | 1.0.28 |
| Ultralytics | 8.3.x |
| OpenCV | 5.0.0 |
| GPU | NVIDIA GeForce RTX 5060 Laptop, CUDA 12.8 |

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate        Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

> The scripts are Windows-first (they were developed there) but contain no hard-coded
> machine paths any more; every path is resolved relative to the repository root. Run all
> commands **from the repository root**.

## 5. Data and weights

Neither the dataset nor the trained weights are stored in this repository:

* **Data** — self-collected from a vending-cabinet platform; see [`data/README.md`](data/README.md)
  for the exact folder layout the scripts expect and how to request access.
* **Weights** — see [`weights/README.md`](weights/README.md). Two files are needed:
  the YOLO11m detector and the ConvNeXt V2-Atto classifier.

Once both are in place, the tree looks like:

```
data/train_images/   3200 scene images (+ YOLO labels)
data/val_images/      680 scene images
data/test/            680 scene images (85 per class)
runs/detect/train_8cls_gpu2/weights/best.pt
results/classifier_imagenet_official/best_model_imagenet.pth
```

## 6. Reproducing the paper

**Step 1 — train Module A (detector).** 30 epochs, img 640, batch 8, patience 10,
mosaic 1.0, mixup 0.1, AMP off (paper Table II).

```bash
python s03_train_yolo11m.py
```

**Step 2 — build the classification split from GT-ROIs.**

```bash
python s00_crop_gt_train.py        # -> cropped_gt_train_new/
python s01_crop_gt_val_pred.py     # -> cropped_gt_val_pred/
python s02_split_cls_data.py       # -> cls_data_new/train (3,566), cls_data_new/val (746)
```

**Step 3 — train Module C.** ConvNeXt V2-Atto, ImageNet init, <=20 epochs, batch 16,
lr 1e-4 AdamW, 224x224, early-stop patience 5 (best val acc 88.47 % at epoch 9, stopped at 14).

```bash
python s06_train_classifier_imagenet.py
```

**Step 4 — select `tau` and `rho` on the validation split** (never on test).

```bash
python s20_ablation_confidence.py      # paper Fig. 5  (tau in 0.01 ... 0.30)
python s22_ablation_roi_expansion.py   # paper Fig. 6  (rho in 0 ... 0.30)
```

**Step 5 — the frozen configuration, evaluated once on the test split.**

```bash
python s32_final_test_tight_roi.py           # paper Fig. 8 / Table VII-VIII
python s33_final_statistical_analysis.py     # Wilson CI, McNemar, Fig. 9
python s34_efficiency_profile.py             # parameters, FLOPs, latency, FPS
```

**Optional — dataset hygiene reported in Section IV-A.**

```bash
python s40_check_duplicates.py
```

### Expected console/files

`results/final_test_conf_0_01_tight_roi/summary.json` must contain
`"total_images": 680`, `"yolo_only_correct": 16`, `"two_stage_only_correct": 46`;
`results/final_statistical_analysis/statistical_summary.json` must contain
`"p_value_two_sided": 0.000176...`. Any deviation means the data split or the weights differ
from the ones used in the paper.

## 7. Reproducibility notes

* The test split is used **once**. All `*_ablation_*` scripts read the **validation** split.
* Both branches of the comparison consume identical YOLO11m detections (same `best.pt`,
  same `tau`), so no detector-side randomness enters the comparison.
* Cropping follows the exact integer-`xyxy` -> expand -> re-encode-JPEG -> reload path
  recorded in `results/validation_roi_ablation_conf_0_01/summary.json`
  (`"crop_logic": "integer_xyxy_then_expand_save_jpg_then_reload"`), because a small
  mismatch there changes the reported accuracies.
* Dataset-level caveats (single-product scenes, one camera rig, eight beverage categories)
  are discussed in Section V of the paper and are not fixable by code.

## 8. Citation

```bibtex
@article{li2026tightroi,
  title   = {Task-Decoupled Fine-Grained Product Recognition Using Tight Predicted
             ROIs for Automated Vending Cabinets},
  author  = {Li, Yihui and Tang, Shaojun and Shen, Zihan and Ge, Yuhang and Xie, Kai},
  journal = {Manuscript under review},
  year    = {2026}
}
```

## 9. License and contact

Code released under the MIT License (see [`LICENSE`](LICENSE)).
Corresponding author: Kai Xie — 500646@yangtzeu.edu.cn.
For questions about reproducing a specific number, please open an issue with the script
name and the value you obtained.
