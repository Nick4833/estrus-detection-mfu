# REAL-TIME AUTOMATED ESTRUS BEHAVIOR DETECTION AND ANALYSIS FOR NORTHERN THAI FARM: A MULTI-CAMERA, COMPUTER VISION SYSTEM

A dual-camera computer-vision pipeline for automated detection of cattle mounting
(estrus) behavior on a smallholder farm, running in real time on an **NVIDIA Jetson
Orin Nano** edge device. The system uses **YOLO26n** detectors for mounting events and
collar-based individual identification, a two-tier (`possible` / `confirmed`) confirmation
scheme, and cross-camera temporal fusion to combine simultaneous views of the same event.

This repository accompanies the paper:

> **Real-Time Automated Estrus Behavior Detection and Analysis for a Northern Thai Farm:
> A Multi-Camera Computer Vision System** — Wicha, S. *et al.*, submitted to *IEEE Access*, 2026.

It contains the detection pipeline, the review dashboard, the evaluation records, and the
statistics script that reproduces every confidence interval and significance test reported
in the paper.

---

## Repository layout

```
estrus-detection-mfu/
├── cattle_mount_detector_dual_folder_upgrade.py   # main detection pipeline
├── dashboard_server.py                            # Flask backend for the review dashboard
├── dashboard.html                                 # dashboard frontend
├── compute_cis.py                                 # reproduces all CIs / significance tests
├── trackers/
│   └── bytetrack_mount.yaml                        # tracker config (not required by detector early testing)
├── models/                                         # model weights — see "Models" below
├── data/                                           # evaluation records + data dictionary
│   ├── CONTROLLED_RECALL_TEST.csv
│   ├── NIGHTTIME_FALSE_POSITIVE_TEST.csv
│   ├── LONG_VIDEO_TEST.csv
│   ├── DATA_DICTIONARY.md
│   └── LICENSE                                      # CC BY 4.0 (data only)
├── results/
│   └── reproducibility_log.txt                     # saved compute_cis.py output
├── requirements.txt
├── LICENSE                                          # MIT (code)
├── CITATION.cff
└── README.md
```

All scripts are designed to be **run from the repository root**, so the relative
`models/`, `trackers/`, `clips/`, and `logs/` paths resolve correctly.

---

## Installation

Requires Python 3.9+ and, for real-time performance, a CUDA-capable device
(the system was developed and evaluated on a Jetson Orin Nano).

```bash
git clone <repository-url>
cd estrus-detection-mfu
pip install -r requirements.txt
```

Pin `torch` / `torchvision` to the build matching your JetPack / CUDA version. See
`requirements.txt` for how to capture exact tested versions.

## Models

The trained weights are distributed as **GitHub Release assets** (they are excluded from
the git history via `.gitignore`). Download them from the latest release and place them in
`models/`:

```
models/mount_best.pt      # mounting-event detector
models/collar_best.pt            # collar / individual-ID detector
```

Paths can be overridden with `--mount-model` / `--collar-model`.

---

## Usage

### Detection pipeline

Process a pair of camera folders (or single video files) with cross-camera fusion:

```bash
python cattle_mount_detector_dual_folder_upgrade.py \
    --cam1 /path/to/cam1_videos \
    --cam2 /path/to/cam2_videos
```

Single-camera mode: omit `--cam2`. Live RTSP mode: add `--live`. Useful options include
`--mount-conf`, `--collar-conf`, `--fusion-window`, `--mount-imgsz`, and `--mount-augment`
(see `--help` for the full list). Detections are written to `logs/mount_log.csv` and clips
are saved under `clips/` (`confirmed/`, `possible/`, `fused/`).

### Review dashboard

```bash
pip install flask opencv-python
python dashboard_server.py
# then open http://<jetson-ip>:5000
```

The dashboard streams both camera previews, lists logged events, plays back saved clips,
and lets a farm operator confirm true events.

---

## Reproducing the paper's statistics

All confidence intervals and significance tests are reproduced by a single script with a
fixed seed:

```bash
python compute_cis.py
```

This runs a seeded (`seed=42`, 10,000-iteration) nonparametric bootstrap for the reported
proportions and precision/recall/F1, exact Poisson intervals for the false-positive rates,
and an exact McNemar test for single-camera vs. fused recall.

The counts fed into the script are taken from the evaluation records in `data/`. See
[`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md) for the full column reference and for a
provenance table mapping each reported metric to the file and rule used to derive it. A saved
console run is kept in `results/reproducibility_log.txt` as a reproducibility artifact.

> **Note on scope.** The evaluation is a single-farm prototype with small sample sizes; the
> paper reports this openly, including negative findings. The `data/` records contain derived
> detection logs and manual scoring only — **raw farm video is withheld for privacy reasons.**

---
