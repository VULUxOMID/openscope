# OpenScope v1 build plan

Six weeks. Four Devin agents work in parallel. Each owns one folder and its tests, so their changes do not collide. Every week ends with merged, tested code and a short public build log.

| Week | Work | Done when |
| --- | --- | --- |
| 1 | Data pipeline and baseline: download with checksums, tiling, tissue mask, patch labels, hard-negative mining, a first ResNet/MobileNet baseline | Patches and labels regenerate with one command; baseline AUC reported |
| 2 | Teacher: fine-tune Owkin Phikon on CAMELYON patches with stain augmentation | Slide-level AUC and FROC on the CAMELYON16 test set, with heatmaps |
| 3 | Student: distill into MobileNetV3 | Student within 2 AUC points of the teacher |
| 4 | Edge: ONNX, TensorRT INT8 with calibration, Pi 5 fallback, benchmark command | Under 1 s per field on an Orin Nano, measured by `bench` |
| 5 | Capture app and review dashboard | Live heatmap on a camera feed; a reviewer can confirm or reject flagged fields |
| 6 | Hardening, docs, reproducible release | A new user reproduces the benchmark from a clean machine with one command |

## Agent lanes

1. **Data:** `openscope/data`, `openscope/stain`
2. **Model:** `openscope/train`, `openscope/eval`
3. **Edge:** `openscope/edge`, `openscope/bench`
4. **App:** `openscope/capture`, `openscope/review`

## Ground rules

- Public, de-identified datasets only. No patient data in this repository.
- No diagnostic claims. Results are research benchmarks.
- Every flagged field goes to a qualified human reviewer.
- Every number in the README comes from a command in this repository that anyone can rerun.
