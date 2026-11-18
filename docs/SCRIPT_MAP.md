# Script map

Every script was renamed for reviewability. The header comment inside each file also records
its original name. All commands are meant to be run from the repository root.

Legend for the *Paper* column: **Fig. n** / **Table n** refer to the artefacts in the submitted
manuscript; **—** means supporting or diagnostic code that does not directly appear in the paper.

## Data preparation

| Script | Original name | Purpose | Paper | Output |
|---|---|---|---|---|
| `s00_crop_gt_train.py` | `crop_gt_train.py` | Crop GT-ROI training images from YOLO labels | Table I | `cropped_gt_train_new/` |
| `s01_crop_gt_val_pred.py` | `crop_gt_val_pred.py` | Crop GT-ROI validation crops | Table I | `cropped_gt_val_pred/` |
| `s02_split_cls_data.py` | `split_cls_data.py` | Assemble `cls_data_new/{train,val}` from the GT crops | Table I | `cls_data_new/` |

## Module A — detection

| Script | Original name | Purpose | Paper | Output |
|---|---|---|---|---|
| `s03_train_yolo11m.py` | `train_yolo_我的GPU版.py` | Train YOLO11m (30 ep, 640, bs 8, patience 10) | Table II, Fig. 3 | `runs/detect/train_8cls_gpu2/` |
| `s03b_train_yolo_stub.py` | `train.py` | Minimal YOLO training stub used for quick trials | — | `runs/detect/` |

## Module B — tight Pred-ROI

| Script | Original name | Purpose | Paper | Output |
|---|---|---|---|---|
| `s04_crop_pred_final.py` | `crop_pred_final.py` | Crop tight Pred-ROI with the final locked settings | Fig. 1, Fig. 8 | `cropped_pred_final_我的GPU版/` |
| `s05_crop_pred_adaptive.py` | `crop_pred_adaptive.py` | Adaptive Pred-ROI cropping variant (explored, not adopted) | — | `cropped_pred_adaptive_我的GPU版/` |

## Module C — classification

| Script | Original name | Purpose | Paper | Output |
|---|---|---|---|---|
| `s06_train_classifier_imagenet.py` | `train_classifier_ImageNet正式版.py` | Train ConvNeXt V2-Atto, official ImageNet init | Fig. 7, Table II | `results/classifier_imagenet_official/` |
| `s07_train_classifier_gpu.py` | `train_classifier_我的GPU版.py` | Same, local GPU run | — | `results/classifier_我的GPU版/` |
| `s08_train_classifier_legacy.py` | `train_classifier.py` | Original (legacy) training script | — | `best_classifier_new.pth` |

## End-to-end evaluation

| Script | Original name | Purpose | Paper | Output |
|---|---|---|---|---|
| `s10_eval_end2end_imagenet.py` | `eval_end2end_ImageNet正式版.py` | End-to-end evaluation with the ImageNet-initialised classifier | — | `results/end2end_imagenet_official/` |
| `s11_eval_end2end_gpu.py` | `eval_end2end_我的GPU版.py` | End-to-end evaluation, local GPU run | — | `results/end2end_我的GPU版/` |
| `s12_eval_end2end_gtroi_legacy.py` | `eval_end2end.py` | Legacy evaluation on GT-ROI crops | — | `results/` |
| `s13_eval_end2end_adaptive.py` | `eval_end2end_adaptive.py` | Evaluation with adaptive ROI expansion | — | `results/end2end_adaptive_我的GPU版/` |

## Ablations

| Script | Original name | Purpose | Paper | Output |
|---|---|---|---|---|
| `s20_ablation_confidence.py` | `检测置信度阈值消融_严格复现正式裁剪.py` | Detection-confidence ablation, `tau` = 0.01 … 0.30 | **Fig. 5**, Table III | `results/confidence_threshold_ablation_exact_crop/` |
| `s21_ablation_confidence_imagenet.py` | `检测置信度阈值消融_官方ImageNet.py` | Same with the ImageNet-init classifier | — | `results/confidence_threshold_ablation_imagenet_official/` |
| `s22_ablation_roi_expansion.py` | `验证集ROI扩展消融_固定阈值0.01.py` | ROI expansion ablation at `tau` = 0.01, `rho` = 0 … 0.30 | **Fig. 6**, Table IV | `results/validation_roi_ablation_conf_0_01/` |
| `s23_ablation_roi_expansion_alt.py` | `ROI扩展比例消融_阈值0.30_严格复现.py` | ROI expansion ablation through the alternative crop pipeline | cross-check | `results/roi_expansion_ablation_conf_0_30_exact_crop/` |
| `s24_ablation_joint.py` | `验证集_检测阈值与ROI联合消融_严格复现.py` | Joint `tau` x `rho` sweep on the validation split | validation behind Table III-IV | `results/validation_joint_ablation_exact_crop/` |
| `s25_ablation_low_threshold_boundary.py` | `验证集低阈值边界检查_修正版.py` | Boundary sanity check below `tau` = 0.01 | — | `results/validation_low_threshold_ablation_corrected/` |

## Final comparison, statistics, efficiency

| Script | Original name | Purpose | Paper | Output |
|---|---|---|---|---|
| `s30_baseline_single_stage.py` | `运行单阶段YOLO11基线.py` | Run the single-stage YOLO11m baseline | Table VII | `results/single_stage_yolo11_baseline/` |
| `s31_baseline_single_stage_conf030.py` | `单阶段YOLO11公平基线_阈值0.30.py` | Fair single-stage baseline at `tau` = 0.30 | cross-check | `results/single_stage_yolo11_conf_0_30/` |
| `s32_final_test_tight_roi.py` | `最终公平测试_阈值0.01_紧致ROI.py` | **Final paired test** on the independent split, `tau`=0.01, `rho`=0 | **Fig. 8**, Tables VII-VIII | `results/final_test_conf_0_01_tight_roi/` |
| `s33_final_statistical_analysis.py` | `最终统计分析_置信区间_McNemar_混淆矩阵.py` | Wilson CI, exact McNemar, confusion matrices, per-class P/R/F1 | **Fig. 9**, Table VIII | `results/final_statistical_analysis/` |
| `s34_efficiency_profile.py` | `统计模型参数_FLOPs_推理时间.py` | Parameters, MACs/FLOPs, latency, FPS | Table IX | `results/efficiency_profile_final/` |

## Dataset hygiene

| Script | Original name | Purpose | Paper | Output |
|---|---|---|---|---|
| `s40_check_duplicates.py` | `检查训练验证测试集重复与近重复.py` | Exact-hash and perceptual-hash overlap check across splits | Section IV-A | `results/duplicate_check/` |
| `s41_export_near_duplicates.py` | `导出跨集合近重复样本供人工核对.py` | Export suspected near-duplicate pairs for manual review | Section IV-A | `results/duplicate_check/cross_split_review/` |
| `s42_find_test_labels.py` | `查找测试集YOLO标签目录.py` | Locate the YOLO label directory of the test split | — | console |

## Figures and diagnostics

| Script | Original name | Purpose | Paper | Output |
|---|---|---|---|---|
| `s50_make_figure1.py` | `make_figure1_sample_auto.py` | Build the framework / sample montage | Fig. 1, Fig. 2 | `results/figure1_sample/` |
| `s51_predict_single_image.py` | `predict.py` | Single-image prediction (detector + classifier) | — | console |
| `s52_visualize_yolo_results.py` | `visualize_yolo_results.py` | Visualise detections on the test split | Fig. 4 | `yolo_vis/` |
| `s53_confusion_matrix_gt.py` | `confusion_matrix_gt.py` | Confusion matrix on GT-ROI crops | — | `confusion_matrix_gt.png` |
| `s54_confusion_matrix_pred.py` | `confusion_matrix_pred.py` | Confusion matrix on Pred-ROI crops | — | `confusion_matrix_pred.png` |
| `s55_collect_error_samples.py` | `collect_error_samples.py` | Collect misclassified samples | Fig. 4 discussion | `error_samples/` |
| `s56_smoke_test_classifier.py` | `test_imagenet_model.py` | Smoke test: load the classifier, predict one crop | — | console |

## Not shipped

| File | Reason |
|---|---|
| `1.py`, `test.py` | empty scratch files |
| course handout PDF | not ours to redistribute |
| `data/`, `cls_data_new/`, `cropped_*/` | raw + intermediate images, available on request |
| `*.pt`, `*.pth` | model checkpoints, see `weights/README.md` |
| `results/**/saved_jpg_rois*/` | ~15k intermediate ROI jpegs, regenerated by the ablation scripts |
