# OpenScope

Open-source, offline AI that turns a basic lab microscope into a cancer triage assistant.

A small edge box (Jetson Orin Nano, with a Raspberry Pi 5 fallback) reads the microscope camera, tiles each field of view, and highlights regions that look like tumor. A lab tech in a clinic without a pathologist can then send only the suspicious slides for remote review. It runs fully offline: no cloud, and no patient data leaves the clinic.

> **Status: day 1 (20 Nov 2026).** Research prototype. Not a medical device and not for diagnosis. Every flag is a prompt for a qualified pathologist to look, never a result.

## Why

Many regions have very few pathologists, so biopsies can wait weeks for someone to read them. Slide scanners that make digital pathology possible cost far more than a basic microscope. OpenScope aims to bring triage to the microscope a clinic already has.

## v1 scope

Breast cancer metastases in lymph-node sections. Trained and evaluated only on public, de-identified, expert-annotated data: CAMELYON16, CAMELYON17 and PatchCamelyon.

## Pipeline

```
camera frame -> tissue mask -> 256 px tiles -> INT8 student model -> tumor heatmap -> flag the field
```

The flag threshold is tuned for high sensitivity. This is triage, so a missed tumor costs more than a false alarm.

## Repo layout

| Folder | What it does | Status |
| --- | --- | --- |
| `openscope/data` | Dataset download with checksums, OpenSlide tiling at 20x, Otsu tissue mask, labels from the lesion XML polygons, hard-negative mining | Tissue mask and tile grid done, with tests |
| `openscope/stain` | Macenko stain normalization and HED color augmentation, so it holds up across labs and stains | Planned |
| `openscope/train` | Teacher: an open pathology foundation model (Owkin Phikon) fine-tuned on patches. Student: MobileNetV3 distilled from the teacher | Planned |
| `openscope/eval` | CAMELYON16 test set: slide-level AUC and lesion-level FROC using the challenge's scoring method, plus heatmaps | Planned |
| `openscope/edge` | ONNX export, then TensorRT INT8 (calibrated on held-out patches) for the Orin Nano; ONNX Runtime fallback for the Pi 5 | Planned |
| `openscope/capture` | GStreamer feed from a USB or C-mount microscope camera, blur check, live heatmap overlay, flagged-field log | Planned |
| `openscope/review` | FastAPI and web dashboard: a remote pathologist gets a queue of flagged fields, confirms or rejects them, and those become new labels | Planned |
| `openscope/bench` | One command for frames per second, latency and power draw on the device | Planned |

## v1 targets

- Teacher: slide-level AUC of at least 0.95 on the CAMELYON16 test set.
- Student: within 2 points of the teacher.
- Edge: under 1 second per field of view on a Jetson Orin Nano.

## Biggest risk

Scanner slides look different from a microscope camera image. The plan: capture-style augmentation from the start (blur, vignetting, color shift, JPEG artifacts), then validation on real camera captures before anything goes near a clinic.

## Build plan

Six weeks, four Devin agents working in parallel, each in its own folder with its own tests. See [docs/PLAN.md](docs/PLAN.md).

## Quick start

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## License

Apache-2.0. See [LICENSE](LICENSE).
