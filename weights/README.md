# Weights

Two checkpoints are required to reproduce the reported numbers. Neither is committed to this
repository (binary blobs do not belong in a source tree, and reviewers should not have to
clone 50 MB to read the code).

| File | Model | Size | Needed for |
|---|---|---|---|
| `best.pt` | YOLO11m detector (30 epochs, 8 classes) | ~38.7 MB | `s32`, `s33`, `s34`, all ablations |
| `best_model_imagenet.pth` | ConvNeXt V2-Atto, ImageNet init | ~13.0 MB | `s06` output, `s32`, `s33`, `s34` |

## Where they go

```
weights/best.pt                                  <- download / train
weights/best_model_imagenet.pth                  <- download / train
```

then copy or symlink to the paths the scripts expect:

```
runs/detect/train_8cls_gpu2/weights/best.pt
results/classifier_imagenet_official/best_model_imagenet.pth
```

Those two paths are the ones hard-coded in `s32_final_test_tight_roi.py` and recorded in
`results/final_test_conf_0_01_tight_roi/summary.json`, so using them keeps the released code
identical to the code that produced the paper.

## Option A — request from the corresponding author (currently the only way)

The checkpoints are **not** attached to this repository or to a GitHub Release. To obtain them
for verification, write to the corresponding author (Kai Xie, xie_kai2001@163.com) and mention
the manuscript title; requests from reviewers are answered first.

Once they are published as a Release, this section will be updated with the download links.

## Option B — retrain

```bash
python s03_train_yolo11m.py            # -> runs/detect/train_8cls_gpu2/weights/best.pt
python s06_train_classifier_imagenet.py # -> results/classifier_imagenet_official/best_model_imagenet.pth
```

Retraining will not reproduce the checkpoint bit-for-bit (GPU/cuDNN non-determinism), so the
final accuracies may differ in the second decimal. Use Option A to verify the paper's numbers.

## Option C — Git LFS

Both files are under GitHub's 100 MB per-file limit, so LFS is optional:

```bash
git lfs install
git add weights/*.pt weights/*.pth
```

`.gitattributes` already routes `*.pt` and `*.pth` through LFS.

## Integrity

If the checkpoints are later published as a Release, the SHA-256 of each file will be listed
next to the download link:

```
# <sha256>  best.pt
# <sha256>  best_model_imagenet.pth
```
