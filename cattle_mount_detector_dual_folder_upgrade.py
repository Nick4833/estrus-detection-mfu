
from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import deque
from dataclasses import dataclass, field
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


MOUNT_MODEL_PATH  = "models/mount_best.pt"
COLLAR_MODEL_PATH = "models/collar_best.pt"

MOUNT_CONF  = 0.5
COLLAR_CONF = 0.1

DEVICE      = "cuda" if __import__("torch").cuda.is_available() else "cpu"
MOUNT_IMGSZ = 640  
USE_HALF    = DEVICE == "cuda"  
MOUNT_AUGMENT = False
COLLAR_CLASSES = {"blue_collar", "green_collar", "red_collar", "yellow_collar"}
HEAD_CLASS     = "head"


CLASS_CONF = {
    "mount-event": 0.8,
    "mountee":     0.5,
    "mounter":     0.8,
}

CLASS_CONF_POSSIBLE = {
    "mount-event": 0.8,
    "mounter":     0.8,
}

ROI_1 = (0.0, 0.0, 1.0, 1.0)
ROI_2 = (0.0, 0.0, 1.0, 1.0)

MOUNT_CONFIRM_SEC      = 0.2
MOUNT_WINDOW_SEC       = 1.0   
MOUNTEE_CONFIRM_FRAMES = 4     
COOLDOWN_SEC           = 30.0

CLIP_PRE_BUFFER_SEC  = 5
CLIP_POST_BUFFER_SEC = 5

FUSION_WINDOW_SEC = 15.0

ANIMAL_BOX_SHRINK = 0

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".mts", ".m4v"}

CLIPS_DIR            = Path("clips")
CLIPS_CONFIRMED_DIR  = CLIPS_DIR / "confirmed"
CLIPS_POSSIBLE_DIR   = CLIPS_DIR / "possible"
LOGS_DIR             = Path("logs")
LOG_FILE             = LOGS_DIR / "mount_log.csv"





logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


for _d in (CLIPS_CONFIRMED_DIR, CLIPS_POSSIBLE_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class MountEvent:
    """Holds everything about one detected mounting event from a single camera."""
    
    cam:              str   = ""      
    video_path:       str   = ""
    timestamp:        str   = ""       
    wall_epoch:       float = 0.0      

    
    event_type:       str   = ""      
    mount_event_conf: float = 0.0
    mounter_conf:     float = 0.0
    mountee_conf:     float = 0.0

    
    mounter_id:       str         = "N/A"
    mounter_id_conf:  float | str = "N/A"
    mounter_id_method: str        = ""
    mountee_id:       str         = "N/A"
    mountee_id_conf:  float | str = "N/A"
    mountee_id_method: str        = ""

    
    clip_name:        str   = ""
    clip_path:        Path  = field(default_factory=Path)
    frames:           list  = field(default_factory=list)   
    
    trigger_dets:     dict  = field(default_factory=dict)   
    trigger_raw_frames: list = field(default_factory=list)  
    best_mountee_det: dict | None = None  
    best_mountee_det: dict | None = None  
    fps:              float = 25.0

    
    fused:            bool  = False
    fused_with_clip:  str   = ""
    
    cam1_mount_event_conf: float = 0.0
    cam1_mounter_conf:     float = 0.0
    cam1_mountee_conf:     float = 0.0
    cam2_mount_event_conf: float = 0.0
    cam2_mounter_conf:     float = 0.0
    cam2_mountee_conf:     float = 0.0
    
    cam1_mounter_id:        str         = ""
    cam1_mounter_id_conf:   float | str = ""
    cam1_mountee_id:        str         = ""
    cam1_mountee_id_conf:   float | str = ""
    cam2_mounter_id:        str         = ""
    cam2_mounter_id_conf:   float | str = ""
    cam2_mountee_id:        str         = ""
    cam2_mountee_id_conf:   float | str = ""
    
    possible_mounter:       str         = ""
    possible_mountee:       str         = ""

    @property
    def mean_det_conf(self) -> float:
        """Aggregate confidence across the three detection classes."""
        vals = [self.mount_event_conf, self.mounter_conf]
        if self.mountee_conf > 0:
            vals.append(self.mountee_conf)
        return sum(vals) / len(vals)

    def collar_conf_float(self, role: str) -> float:
        """Return numeric collar confidence for 'mounter' or 'mountee', or 0."""
        v = self.mounter_id_conf if role == "mounter" else self.mountee_id_conf
        return float(v) if isinstance(v, float) else 0.0






CSV_HEADER = [
    "timestamp", "cam", "video_source", "event_type",
    "mount_event_conf", "mounter_conf", "mountee_conf",
    "cam1_mount_event_conf", "cam1_mounter_conf", "cam1_mountee_conf",
    "cam2_mount_event_conf", "cam2_mounter_conf", "cam2_mountee_conf",
    "mounter_id", "mounter_id_conf", "mounter_id_method",
    "mountee_id", "mountee_id_conf", "mountee_id_method",
    "cam1_mounter_id", "cam1_mounter_id_conf",
    "cam1_mountee_id", "cam1_mountee_id_conf",
    "cam2_mounter_id", "cam2_mounter_id_conf",
    "cam2_mountee_id", "cam2_mountee_id_conf",
    "possible_mounter", "possible_mountee",
    "clip_filename",
    "fused", "fused_with_clip",
]


def _fmt_conf(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def write_csv(events: list[MountEvent]) -> None:
    """Append events to the CSV log — each run adds to the existing file."""
    file_exists = LOG_FILE.exists() and LOG_FILE.stat().st_size > 0
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()  
        for ev in events:
            writer.writerow({
                "timestamp":          ev.timestamp,
                "cam":                ev.cam,
                "video_source":       ev.video_path,
                "event_type":         ev.event_type,
                "mount_event_conf":   f"{ev.mount_event_conf:.3f}",
                "mounter_conf":       f"{ev.mounter_conf:.3f}",
                "mountee_conf":       f"{ev.mountee_conf:.3f}",
                
                "cam1_mount_event_conf": f"{ev.cam1_mount_event_conf:.3f}" if ev.fused else (f"{ev.mount_event_conf:.3f}" if "CAM1" in ev.cam else ""),
                "cam1_mounter_conf":     f"{ev.cam1_mounter_conf:.3f}"     if ev.fused else (f"{ev.mounter_conf:.3f}"     if "CAM1" in ev.cam else ""),
                "cam1_mountee_conf":     f"{ev.cam1_mountee_conf:.3f}"     if ev.fused else (f"{ev.mountee_conf:.3f}"     if "CAM1" in ev.cam else ""),
                "cam2_mount_event_conf": f"{ev.cam2_mount_event_conf:.3f}" if ev.fused else (f"{ev.mount_event_conf:.3f}" if "CAM2" in ev.cam else ""),
                "cam2_mounter_conf":     f"{ev.cam2_mounter_conf:.3f}"     if ev.fused else (f"{ev.mounter_conf:.3f}"     if "CAM2" in ev.cam else ""),
                "cam2_mountee_conf":     f"{ev.cam2_mountee_conf:.3f}"     if ev.fused else (f"{ev.mountee_conf:.3f}"     if "CAM2" in ev.cam else ""),
                "mounter_id":         ev.mounter_id,
                "mounter_id_conf":    _fmt_conf(ev.mounter_id_conf),
                "mounter_id_method":  ev.mounter_id_method,
                "mountee_id":         ev.mountee_id,
                "mountee_id_conf":    _fmt_conf(ev.mountee_id_conf),
                "mountee_id_method":  ev.mountee_id_method,
                "cam1_mounter_id":      ev.cam1_mounter_id,
                "cam1_mounter_id_conf": _fmt_conf(ev.cam1_mounter_id_conf) if ev.cam1_mounter_id_conf != "" else "",
                "cam1_mountee_id":      ev.cam1_mountee_id,
                "cam1_mountee_id_conf": _fmt_conf(ev.cam1_mountee_id_conf) if ev.cam1_mountee_id_conf != "" else "",
                "cam2_mounter_id":      ev.cam2_mounter_id,
                "cam2_mounter_id_conf": _fmt_conf(ev.cam2_mounter_id_conf) if ev.cam2_mounter_id_conf != "" else "",
                "cam2_mountee_id":      ev.cam2_mountee_id,
                "cam2_mountee_id_conf": _fmt_conf(ev.cam2_mountee_id_conf) if ev.cam2_mountee_id_conf != "" else "",
                "possible_mounter":   ev.possible_mounter,
                "possible_mountee":   ev.possible_mountee,
                "clip_filename":      ev.clip_name,
                "fused":              str(ev.fused),
                "fused_with_clip":    ev.fused_with_clip,
            })
    log.info("CSV written → %s  (%d events)", LOG_FILE, len(events))





def write_clip(
    frames_v1: list,
    fps: float,
    clip_path: Path,
    frames_v2: list | None = None,
) -> None:
    """Write annotated clip, optionally side-by-side with a second angle."""
    if not frames_v1:
        log.warning("write_clip: no frames — skipping %s", clip_path.name)
        return

    h, w = frames_v1[0][0].shape[:2]

    if frames_v2:
        max_len = max(len(frames_v1), len(frames_v2))
        black   = np.zeros((h, w, 3), dtype=np.uint8)
        while len(frames_v1) < max_len:
            frames_v1.append((black.copy(), ""))
        while len(frames_v2) < max_len:
            frames_v2.append((black.copy(), ""))
        out_w = w * 2
    else:
        out_w = w

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (out_w, h))

    for i, (f1, _) in enumerate(frames_v1):
        if frames_v2:
            f2 = frames_v2[i][0]
            if f2.shape[:2] != (h, w):
                f2 = cv2.resize(f2, (w, h))
            combined = np.hstack([f1, f2])
        else:
            combined = f1
        writer.write(combined)

    writer.release()
    log.info("Clip saved → %s  (%d frames @ %.1f fps)",
             clip_path.name, len(frames_v1), fps)





def _colour_for_class(class_name: str) -> tuple[int, int, int]:
    palette = {
        "mount-event": (0,   255, 128),
        "mounter":     (0,   200, 255),
        "mountee":     (255, 180,   0),
    }
    return palette.get(class_name, (200, 200, 200))


def annotate_frame(
    frame: np.ndarray,
    detections: list[dict],
    cam_label: str,
    timestamp: str,
    roi: tuple[float, float, float, float],
    collar_dets: list[dict] | None = None,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    cv2.rectangle(out,
                  (int(roi[0] * w), int(roi[1] * h)),
                  (int(roi[2] * w), int(roi[3] * h)),
                  (80, 80, 80), 1)

    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["xyxy"])
        colour = _colour_for_class(det["class_name"])
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        text = f"{det['class_name']} {det['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 2, y1), colour, -1)
        cv2.putText(out, text, (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    COLLAR_COLOUR = (255, 255, 0)
    for det in (collar_dets or []):
        x1, y1, x2, y2 = (int(v) for v in det["xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), COLLAR_COLOUR, 3)
        text = (f"{det['class_name']} {det['conf']:.2f}"
                if det["conf"] else det["class_name"])
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        lx, ly = x1, y2 + th + 4
        cv2.rectangle(out, (lx, ly - th - 4), (lx + tw + 2, ly), COLLAR_COLOUR, -1)
        cv2.putText(out, text, (lx, ly - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(out, cam_label, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, timestamp, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return out







MOUNT_TRACKER_CFG = str(Path(__file__).parent / "trackers" / "bytetrack_mount.yaml")


def run_mount_tracking(
    frame: np.ndarray,
    model: YOLO,
    roi: tuple[float, float, float, float],
    conf_map: dict[str, float] | None = None,
    persist: bool = True,
    _gray_cache: list | None = None,
) -> dict[str, list[dict]]:
    """
    Run YOLO tracking on a single frame and return detections filtered by
    conf_map thresholds and ROI.  Each detection includes a 'track_id' field
    (int) when the tracker assigns one, or -1 for untracked boxes.

    Uses a custom ByteTrack config (trackers/bytetrack_mount.yaml) tuned for
    intermittent low-confidence classes (notably "mount-event").  Pass
    persist=True so the tracker state carries across consecutive calls on
    the same video stream.

    NOTE: prefer run_mount_tracking_multi() when you need both a "confirmed"
    and a "possible" threshold view of the same frame — it runs the model
    once and shares track IDs between the two views, instead of resetting
    the tracker mid-frame.
    """
    thresholds = conf_map if conf_map is not None else CLASS_CONF
    
    if _gray_cache is not None and _gray_cache[0] is not None:
        gray_bgr = _gray_cache[0]
    else:
        gray_bgr = cv2.cvtColor(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        if _gray_cache is not None:
            _gray_cache[0] = gray_bgr
    results = model.track(
        source=gray_bgr,
        conf=min(thresholds.values()),
        persist=persist,
        tracker=MOUNT_TRACKER_CFG,
        iou=0.5,
        imgsz=MOUNT_IMGSZ,
        half=USE_HALF,
        augment=MOUNT_AUGMENT,
        device=DEVICE,
        verbose=False,
    )
    found: dict[str, list[dict]] = {}
    if not results or results[0].boxes is None:
        return found

    h, w = frame.shape[:2]
    boxes = results[0].boxes
    for i, box in enumerate(boxes):
        cls_name = model.names[int(box.cls[0])]
        conf     = float(box.conf[0])
        xyxy     = tuple(float(v) for v in box.xyxy[0])

        if conf < thresholds.get(cls_name, MOUNT_CONF):
            continue

        cx = (xyxy[0] + xyxy[2]) / 2.0 / w
        cy = (xyxy[1] + xyxy[3]) / 2.0 / h
        if not (roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]):
            continue

        
        track_id = int(boxes.id[i]) if boxes.id is not None else -1

        
        if cls_name in ("mounter", "mountee") and ANIMAL_BOX_SHRINK > 0.0:
            x1, y1, x2, y2 = xyxy
            dx = (x2 - x1) * ANIMAL_BOX_SHRINK
            dy = (y2 - y1) * ANIMAL_BOX_SHRINK
            xyxy = (x1 + dx, y1 + dy, x2 - dx, y2 - dy)

        found.setdefault(cls_name, []).append(
            {"class_name": cls_name, "conf": conf, "xyxy": xyxy, "track_id": track_id}
        )
    return found


def run_mount_tracking_multi(
    frame: np.ndarray,
    model: YOLO,
    roi: tuple[float, float, float, float],
    conf_maps: dict[str, dict[str, float]],
    _gray_cache: list | None = None,
    tracker_slot: list | None = None,  
) -> dict[str, dict[str, list[dict]]]:
    """
    Stateless multi-view inference for the mount model.

    Runs ``model.predict()`` ONCE on the frame and builds multiple filtered
    detection views (one per entry in ``conf_maps``) from the same set of
    boxes.

    Why predict() instead of track()
    ────────────────────────────────
    Earlier versions of this function used ``model.track(persist=True)`` to
    get stable per-detection IDs.  That had two unfixable problems:

    1. Two cameras sharing the same YOLO model share Ultralytics'
       ``predictor.trackers`` slot.  In dual-camera mode each call from
       CAM2 clobbered CAM1's tracker state (and vice versa), so tracks
       never survived a single frame and ``_max_age`` was stuck at 0
       — confirmed empirically: identical detection confidences across
       solo and dual-cam runs, but ages frozen only in dual-cam.
       Per-camera tracker swap-in/swap-out was attempted and didn't
       help, because Ultralytics also re-initialises trackers when input
       dimensions change between calls.

    2. The downstream code never actually needed track IDENTITY — only
       "has this class been detected for n consecutive frames?"  That
       question is answered more directly and reliably by a per-class
       run-length counter on the caller side, with no tracking required.

    So this function now just predicts, and callers track consecutive-
    frame counts themselves.  Detections still carry ``track_id`` for
    API stability, but the value is always -1.

    Returns:
        dict mapping the key of each conf_map to its filtered detection dict
        (same shape as the old run_mount_tracking()'s return value).
    """
    
    if _gray_cache is not None and _gray_cache[0] is not None:
        gray_bgr = _gray_cache[0]
    else:
        gray_bgr = cv2.cvtColor(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        if _gray_cache is not None:
            _gray_cache[0] = gray_bgr

    
    
    
    all_thresholds = [
        v for cmap in conf_maps.values() for v in cmap.values()
    ]
    min_conf = min(all_thresholds) if all_thresholds else MOUNT_CONF

    results = model.predict(
        source=gray_bgr,
        conf=min_conf,
        iou=0.5,
        imgsz=MOUNT_IMGSZ,
        half=USE_HALF,
        augment=MOUNT_AUGMENT,
        device=DEVICE,
        verbose=False,
    )

    
    out: dict[str, dict[str, list[dict]]] = {k: {} for k in conf_maps}

    if not results or results[0].boxes is None:
        return out

    h, w = frame.shape[:2]
    boxes = results[0].boxes

    for i, box in enumerate(boxes):
        cls_name = model.names[int(box.cls[0])]
        conf     = float(box.conf[0])
        xyxy     = tuple(float(v) for v in box.xyxy[0])

        cx = (xyxy[0] + xyxy[2]) / 2.0 / w
        cy = (xyxy[1] + xyxy[3]) / 2.0 / h
        if not (roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]):
            continue

        
        if cls_name in ("mounter", "mountee") and ANIMAL_BOX_SHRINK > 0.0:
            x1, y1, x2, y2 = xyxy
            dx = (x2 - x1) * ANIMAL_BOX_SHRINK
            dy = (y2 - y1) * ANIMAL_BOX_SHRINK
            xyxy = (x1 + dx, y1 + dy, x2 - dx, y2 - dy)

        det = {"class_name": cls_name, "conf": conf, "xyxy": xyxy,
               "track_id": -1}

        
        
        for view_name, cmap in conf_maps.items():
            cutoff = cmap.get(cls_name)
            if cutoff is None:
                continue  
            if conf < cutoff:
                continue
            out[view_name].setdefault(cls_name, []).append(det)

    return out



def run_mount_inference(
    frame: np.ndarray,
    model: YOLO,
    roi: tuple[float, float, float, float],
    conf_map: dict[str, float] | None = None,
) -> dict[str, list[dict]]:
    """Thin wrapper — delegates to run_mount_tracking for backward compat."""
    return run_mount_tracking(frame, model, roi, conf_map)


def _crop_box(
    frame: np.ndarray,
    xyxy: tuple,
    pad: float = 0.05,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    pw = int((x2 - x1) * pad);  ph = int((y2 - y1) * pad)
    x1 = max(0, x1 - pw);       y1 = max(0, y1 - ph)
    x2 = min(w, x2 + pw);       y2 = min(h, y2 + ph)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def _dominant_coat_colour(crop: np.ndarray) -> str:
    if crop.size == 0:
        return "unknown"
    hsv       = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v   = cv2.split(hsv)
    animal_mask = ~((s < 40) & (v > 200))
    if animal_mask.sum() < 100:
        return "unknown"
    mean_v = float(v[animal_mask].mean())
    mean_s = float(s[animal_mask].mean())
    mean_h = float(h[animal_mask].mean())
    if mean_v < 60:
        return "black"
    if mean_v > 180 and mean_s < 40:
        return "white"
    if mean_s > 50 and 5 <= mean_h <= 25:
        return "brown"
    if mean_v > 100 and mean_s < 40:
        return "grey"
    return "mixed"


def _quick_collar_label(
    frame: np.ndarray,
    animal_xyxy: tuple,
    collar_model: YOLO,
) -> str | None:
    """Quick single-frame collar label using head-guided detection."""
    cls_name, _, _, _ = _detect_head_and_collar(frame, animal_xyxy, collar_model)
    return cls_name.lower() if cls_name else None


def _coat_colour_only(
    frame: np.ndarray,
    animal_xyxy: tuple,
    collar_model: YOLO,
) -> tuple[str, float | str, str, None]:
    """Get coat colour using head crop from collar+head model."""
    _, _, _, head_abs = _detect_head_and_collar(frame, animal_xyxy, collar_model)
    if head_abs is not None:
        head_crop, _ = _crop_box(frame, head_abs)
        if head_crop.size > 0:
            return _dominant_coat_colour(head_crop), "N/A", "coat_colour", None
    crop, _ = _crop_box(frame, animal_xyxy)
    if crop.size == 0:
        return "N/A", "N/A", "coat_colour", None
    return _dominant_coat_colour(crop), "N/A", "coat_colour", None


def _detect_head_and_collar(
    frame: np.ndarray,
    animal_xyxy: tuple,
    collar_model: YOLO,
) -> tuple[str | None, float, tuple | None, tuple | None]:
    """
    Crop to animal bbox, run the collar+head model, find the head detection,
    then find a collar within or near the head bbox.

    Returns (collar_class, conf, collar_xyxy_abs, head_xyxy_abs)
    or (None, 0.0, None, None) if nothing found.

    Strategy:
      1. Crop to animal bbox
      2. Run collar+head model on crop
      3. Find head box — use as the region of interest for collar
      4. Find best collar whose centre falls inside the head bbox
      5. Fallback: best collar anywhere in the animal crop
    """
    crop, (ox, oy, _, _) = _crop_box(frame, animal_xyxy)
    if crop.size == 0:
        return None, 0.0, None, None

    try:
        results = collar_model.predict(
            source=crop, conf=COLLAR_CONF, verbose=False,
            half=USE_HALF, device=DEVICE,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None, 0.0, None, None

        heads   = []
        collars = []
        for box in results[0].boxes:
            cls_name = collar_model.names[int(box.cls[0])]
            conf     = float(box.conf[0])
            bx1, by1, bx2, by2 = (float(v) for v in box.xyxy[0])
            det = {"class_name": cls_name, "conf": conf,
                   "xyxy": (bx1, by1, bx2, by2)}
            if cls_name == HEAD_CLASS:
                heads.append(det)
            elif cls_name in COLLAR_CLASSES:
                collars.append(det)

        log.debug("_detect_head_and_collar: crop=%dx%d  heads=%d  collars=%d",
                  crop.shape[1], crop.shape[0], len(heads), len(collars))

        if not collars:
            return None, 0.0, None, None

        
        best_head = max(heads, key=lambda d: d["conf"]) if heads else None
        head_abs  = None
        if best_head:
            hx1, hy1, hx2, hy2 = best_head["xyxy"]
            head_abs = (ox + hx1, oy + hy1, ox + hx2, oy + hy2)

        
        winner = None
        if best_head:
            hx1, hy1, hx2, hy2 = best_head["xyxy"]
            inside = [c for c in collars
                      if hx1 <= (c["xyxy"][0] + c["xyxy"][2]) / 2 <= hx2
                      and hy1 <= (c["xyxy"][1] + c["xyxy"][3]) / 2 <= hy2]
            if inside:
                winner = max(inside, key=lambda d: d["conf"])

        
        if winner is None:
            winner = max(collars, key=lambda d: d["conf"])

        cx1, cy1, cx2, cy2 = winner["xyxy"]
        collar_abs = (ox + cx1, oy + cy1, ox + cx2, oy + cy2)
        return winner["class_name"], winner["conf"], collar_abs, head_abs

    except Exception as exc:
        log.debug("_detect_head_and_collar error: %s", exc)
        return None, 0.0, None, None


def vote_collar(
    frame: np.ndarray,
    animal_dets: list[dict],
    collar_model: YOLO,
    window_frames: list[np.ndarray] | None = None,
) -> tuple[str, float | str, str, tuple | None]:
    """
    Identify an animal's collar by:
      1. Cropping to the animal bbox
      2. Running the collar+head model to find the head, then the collar within it
      3. Voting across all window_frames — most-voted collar class wins

    Falls back to coat colour if no collar is found across all frames.
    """
    from collections import Counter

    best_det = max(animal_dets, key=lambda d: d["conf"])
    ref_xyxy = best_det["xyxy"]

    collar_votes:     Counter = Counter()
    coat_votes:       Counter = Counter()
    best_collar_xyxy: tuple | None = None
    best_collar_conf: float = 0.0

    frames_to_check = window_frames if window_frames else [frame]

    for f in frames_to_check:
        cls_name, conf, collar_abs, _ = _detect_head_and_collar(f, ref_xyxy, collar_model)
        if cls_name is not None:
            collar_votes[cls_name] += 1
            if conf > best_collar_conf:
                best_collar_conf = conf
                best_collar_xyxy = collar_abs
            continue

        
        try:
            crop, _ = _crop_box(f, ref_xyxy)
            coat_votes[_dominant_coat_colour(crop)] += 1
        except Exception:
            pass

    if collar_votes:
        winner = collar_votes.most_common(1)[0][0]
        log.debug("vote_collar: winner=%s (%d/%d frames)",
                  winner, collar_votes[winner], len(frames_to_check))
        return winner, best_collar_conf, "collar", best_collar_xyxy

    if coat_votes:
        winner = coat_votes.most_common(1)[0][0]
        return winner, "N/A", "coat_colour", None

    return "N/A", "N/A", "N/A", None


def identify_animal(
    frame: np.ndarray,
    animal_xyxy: tuple,
    collar_model: YOLO,
) -> tuple[str, float | str, str, tuple | None]:
    """Single-frame fallback using head-guided collar detection."""
    cls_name, conf, collar_abs, _ = _detect_head_and_collar(
        frame, animal_xyxy, collar_model
    )
    if cls_name is not None:
        return cls_name, conf, "collar", collar_abs

    
    try:
        crop, _ = _crop_box(frame, animal_xyxy)
        return _dominant_coat_colour(crop), "N/A", "coat_colour", None
    except Exception:
        pass

    return "N/A", "N/A", "N/A", None





class VideoProcessor:
    """
    Processes a single video file synchronously.
    Instead of writing CSV rows inline, it returns MountEvent objects
    that the caller collects for the fusion pass.
    """

    def __init__(
        self,
        video_path: str,
        cam_label: str,
        roi: tuple[float, float, float, float],
        mount_model: YOLO,
        collar_model: YOLO,
        video_start_epoch: float = 0.0,
    ) -> None:
        self.video_path        = video_path
        self.cam_label         = cam_label
        self.roi               = roi
        self.mount_model       = mount_model
        self.collar_model      = collar_model
        self.video_start_epoch = video_start_epoch   

        self._is_rtsp = str(video_path).lower().startswith("rtsp://")
        self.cap = cv2.VideoCapture(video_path)
        if self._is_rtsp:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        self.fps          = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_idx    = 0

        pre_buf_frames   = int(CLIP_PRE_BUFFER_SEC * self.fps)
        self._pre_buffer:     deque = deque(maxlen=pre_buf_frames)
        self._raw_pre_buffer: deque = deque(maxlen=pre_buf_frames)

        self._confirm_frames_needed = max(1, round(MOUNT_CONFIRM_SEC * self.fps))
        self._window_frames_total   = max(1, round(MOUNT_WINDOW_SEC * self.fps))

        
        
        
        self._confirmed_cooccur_frames: int = 0
        self._possible_cooccur_frames:  int = 0

        
        self._in_window:           bool = False
        self._window_frames_left:  int  = 0
        self._mountee_upgrade_count: int = 0   

        
        self._inference_times: list[float] = []

        self._cooldown_frames_total = round(COOLDOWN_SEC * self.fps)
        self._cooldown_frames_left  = 0

        self._recording             = False
        self._post_frames_remaining = 0
        self._current_event: MountEvent | None = None
        self._peak_mountee_conf:    float = 0.0   
        self._peak_mountee_conf:    float = 0.0   

        
        self.events: list[MountEvent] = []

        log.info("[%s] Opened %s  (%.1f fps, %d frames)",
                 cam_label, video_path, self.fps, self.total_frames)

    

    def read_frame(self) -> tuple[np.ndarray | None, str, float]:
        """Return (frame, ts_str, wall_epoch) or (None, '', 0) at EOF."""
        ret, frame = self.cap.read()
        if not ret:
            return None, "", 0.0
        if self._is_rtsp:
            
            wall_epoch = time.time()
            ts_str     = f"[{self.cam_label}] frame {self.frame_idx:06d}  {datetime.fromtimestamp(wall_epoch).strftime("%H:%M:%S.%f")[:-3]}"
        else:
            pos_sec    = self.frame_idx / self.fps
            wall_epoch = self.video_start_epoch + pos_sec
            ts_str     = f"[{self.cam_label}] frame {self.frame_idx:06d}  +{pos_sec:.2f}s"
        self.frame_idx += 1
        return frame, ts_str, wall_epoch

    

    def process_frame(
        self,
        frame: np.ndarray,
        ts_str: str,
        wall_epoch: float,
    ) -> None:
        """
        Simplified tracking-based pipeline — no flicker phase.

        Phases:
          pre-window  → accumulating track ages + mountee counts
          window      → waiting for mountee to confirm; extends while trigger active
          recording   → post-buffer frames being collected
          cooldown    → quiet period after an event

        The window opens as soon as a trigger track (mounter, mount-event, or
        mounter+mountee pair) reaches age >= _confirm_frames_needed.  Mountee
        detections seen *before* the window opens are counted immediately so
        events that peak early are not missed.  The window extends (resets
        _window_frames_left) each frame the trigger is still active.
        """
        wall_ts = datetime.fromtimestamp(wall_epoch).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        
        
        
        
        
        
        
        
        
        
        _t0 = time.perf_counter()
        _gray = [None]
        _views = run_mount_tracking_multi(
            frame, self.mount_model, self.roi,
            conf_maps={"possible": CLASS_CONF_POSSIBLE,
                       "confirmed": CLASS_CONF},
            _gray_cache=_gray,
        )
        mount_dets_possible  = _views["possible"]
        mount_dets_confirmed = _views["confirmed"]
        self._inference_times.append(time.perf_counter() - _t0)

        
        has_mount_event_possible = "mount-event" in mount_dets_possible
        has_mounter_possible     = "mounter"     in mount_dets_possible
        has_mount_event_conf     = "mount-event" in mount_dets_confirmed
        has_mounter_conf         = "mounter"     in mount_dets_confirmed
        has_mountee_conf         = "mountee"     in mount_dets_confirmed

        
        
        
        if has_mount_event_conf and has_mounter_conf and has_mountee_conf:
            self._confirmed_cooccur_frames += 1
        else:
            self._confirmed_cooccur_frames = 0

        
        if has_mount_event_possible and has_mounter_possible:
            self._possible_cooccur_frames += 1
        else:
            self._possible_cooccur_frames = 0

        n = self._confirm_frames_needed
        trigger_confirmed = self._confirmed_cooccur_frames >= n
        
        trigger_possible  = (self._possible_cooccur_frames >= n
                             and self._confirmed_cooccur_frames == 0)

        
        all_mount_dets = {**mount_dets_confirmed}
        for cls, dets in mount_dets_possible.items():
            all_mount_dets.setdefault(cls, dets)
        det_summary = "  ".join(
            f"{cls}={max(d['conf'] for d in dets):.2f}"
            for cls, dets in sorted(all_mount_dets.items())
        ) if all_mount_dets else "no detections"

        phase = (
            f"recording(upgrade={self._mountee_upgrade_count}/{MOUNTEE_CONFIRM_FRAMES} win={self._window_frames_left}f)"
                            if self._recording and self._in_window else
            "recording"     if self._recording else
            f"cooldown({self._cooldown_frames_left}f)"
                            if self._cooldown_frames_left > 0 else
            f"pre(conf={self._confirmed_cooccur_frames} pos={self._possible_cooccur_frames}/{n})"
        )
        _fps_inst = (1.0 / self._inference_times[-1]
                     if self._inference_times and self._inference_times[-1] > 0 else 0.0)
        _frame_ts = datetime.fromtimestamp(wall_epoch).strftime("%H:%M:%S.%f")[:-3]

        log.debug(
            "[%s] triggers: confirmed=%s possible=%s  cooccur(conf=%d pos=%d)  n=%d",
            self.cam_label, trigger_confirmed, trigger_possible,
            self._confirmed_cooccur_frames, self._possible_cooccur_frames, n)
        log.debug("[%s] frame=%06d  %s  phase=%-40s  %5.1f fps  %s",
                  self.cam_label, self.frame_idx - 1, _frame_ts, phase, _fps_inst, det_summary)

        all_dets  = [d for dets in all_mount_dets.values() for d in dets]
        ann_frame = annotate_frame(frame, all_dets, self.cam_label, ts_str, self.roi)

        

        if self._recording and self._current_event is not None:
            self._current_event.frames.append((ann_frame, ts_str))
            self._raw_pre_buffer.append(frame)

            
            if has_mountee_conf:
                mc = max(d["conf"] for d in mount_dets_confirmed["mountee"])
                if mc > self._peak_mountee_conf:
                    self._peak_mountee_conf = mc
                    self._current_event.best_mountee_det = max(
                        mount_dets_confirmed["mountee"], key=lambda d: d["conf"]
                    )

            
            if has_mountee_conf:
                mc = max(d["conf"] for d in mount_dets_confirmed["mountee"])
                if mc > self._peak_mountee_conf:
                    self._peak_mountee_conf = mc
                    self._current_event.best_mountee_det = max(
                        mount_dets_confirmed["mountee"], key=lambda d: d["conf"]
                    )

            
            
            if self._current_event.event_type == "possible":
                if trigger_confirmed:
                    log.info("[%s] Confirmed trigger fired during recording → upgrading to CONFIRMED.",
                             self.cam_label)
                    self._in_window             = False
                    self._mountee_upgrade_count = 0
                    self._current_event.event_type = "confirmed"
                    self._current_event.clip_path  = (
                        CLIPS_CONFIRMED_DIR /
                        self._current_event.clip_path.name.replace(
                            "mount_possible_", "mount_confirmed_"))
                    self._current_event.clip_name  = self._current_event.clip_path.name
                elif self._in_window:
                    self._window_frames_left -= 1
                    if has_mountee_conf and (has_mount_event_conf or has_mounter_conf):
                        self._mountee_upgrade_count += 1
                    if self._mountee_upgrade_count >= MOUNTEE_CONFIRM_FRAMES:
                        log.info("[%s] Mountee confirmed (%d/%d) → upgrading to CONFIRMED.",
                                 self.cam_label, self._mountee_upgrade_count, MOUNTEE_CONFIRM_FRAMES)
                        self._in_window             = False
                        self._mountee_upgrade_count = 0
                        self._current_event.event_type = "confirmed"
                        self._current_event.clip_path  = (
                            CLIPS_CONFIRMED_DIR /
                            self._current_event.clip_path.name.replace(
                                "mount_possible_", "mount_confirmed_"))
                        self._current_event.clip_name  = self._current_event.clip_path.name
                    elif self._window_frames_left <= 0:
                        log.info("[%s] Upgrade window expired — staying POSSIBLE.", self.cam_label)
                        self._in_window             = False
                        self._mountee_upgrade_count = 0

            self._post_frames_remaining -= 1
            if self._post_frames_remaining <= 0:
                self._finalise()

        elif self._cooldown_frames_left > 0:
            self._cooldown_frames_left -= 1
            self._confirmed_cooccur_frames = 0
            self._possible_cooccur_frames  = 0
            self._in_window                = False
            self._mountee_upgrade_count    = 0
            self._pre_buffer.append((ann_frame, ts_str))
            self._raw_pre_buffer.append(frame)
            if self._cooldown_frames_left == 0:
                log.info("[%s] Cooldown finished.", self.cam_label)

        elif trigger_confirmed:
            log.info("[%s] MOUNT EVENT [CONFIRMED] @ %s", self.cam_label, wall_ts)
            self._trigger_event(frame, ann_frame, mount_dets_confirmed,
                                wall_ts, wall_epoch, event_type="confirmed")

        elif trigger_possible:
            log.info("[%s] MOUNT EVENT [POSSIBLE] @ %s — opening upgrade window.",
                     self.cam_label, wall_ts)
            self._in_window              = True
            self._window_frames_left     = self._window_frames_total
            self._mountee_upgrade_count  = 0
            self._confirmed_cooccur_frames = 0
            self._possible_cooccur_frames  = 0
            self._trigger_event(frame, ann_frame, mount_dets_possible,
                                wall_ts, wall_epoch, event_type="possible")

        else:
            self._pre_buffer.append((ann_frame, ts_str))
            self._raw_pre_buffer.append(frame)

    

    def _trigger_event(
        self,
        raw_frame: np.ndarray,
        ann_frame: np.ndarray,
        mount_dets: dict,
        wall_ts: str,
        wall_epoch: float,
        event_type: str = "confirmed",
    ) -> None:
        log.info("[%s] MOUNT EVENT [%s] @ %s", self.cam_label, event_type.upper(), wall_ts)

        has_mount_event = "mount-event" in mount_dets
        has_mounter     = "mounter" in mount_dets
        has_mountee     = "mountee" in mount_dets

        me_conf  = max(d["conf"] for d in mount_dets["mount-event"]) if has_mount_event else 0.0
        mr_conf  = max(d["conf"] for d in mount_dets["mounter"])     if has_mounter     else 0.0
        me2_conf = max(d["conf"] for d in mount_dets["mountee"])     if has_mountee     else 0.0

        mounter_det = max(mount_dets["mounter"], key=lambda d: d["conf"]) if has_mounter else None
        mountee_det = max(mount_dets["mountee"], key=lambda d: d["conf"]) if has_mountee else None

        
        
        mounter_id, mounter_id_conf, mounter_id_method = "N/A", "N/A", "pending"
        mountee_id, mountee_id_conf, mountee_id_method = "N/A", "N/A", "pending"

        all_dets  = [d for dets in mount_dets.values() for d in dets]
        ann_frame = annotate_frame(raw_frame, all_dets, self.cam_label, wall_ts,
                                   self.roi)

        safe_ts   = wall_ts.replace(" ", "_").replace(":", "-")
        clip_name = f"mount_{event_type}_{self.cam_label}_{safe_ts}.mp4"
        clip_dir  = CLIPS_CONFIRMED_DIR if event_type == "confirmed" else CLIPS_POSSIBLE_DIR
        clip_path = clip_dir / clip_name

        ev = MountEvent(
            cam               = self.cam_label,
            video_path        = self.video_path,
            timestamp         = wall_ts,
            wall_epoch        = wall_epoch,
            event_type        = event_type,
            mount_event_conf  = me_conf,
            mounter_conf      = mr_conf,
            mountee_conf      = me2_conf,
            mounter_id        = mounter_id,
            mounter_id_conf   = mounter_id_conf,
            mounter_id_method = mounter_id_method,
            mountee_id        = mountee_id,
            mountee_id_conf   = mountee_id_conf,
            mountee_id_method = mountee_id_method,
            clip_name         = clip_name,
            clip_path         = clip_path,
            fps               = self.fps,
        )

        self._recording             = True
        self._post_frames_remaining = int(CLIP_POST_BUFFER_SEC * self.fps)
        ev.frames                   = list(self._pre_buffer) + [(ann_frame, wall_ts)]
        ev.trigger_dets             = mount_dets
        ev.trigger_raw_frames       = list(self._raw_pre_buffer)
        self._current_event         = ev

    def _finalise(self) -> None:
        """Post-buffer complete — run collar ID across all event frames, write clip."""
        self._recording            = False
        self._cooldown_frames_left = self._cooldown_frames_total
        if self._current_event is not None:
            ev = self._current_event
            self._current_event = None

            
            
            
            
            
            
            
            
            _all_raw = list(ev.trigger_raw_frames) + list(self._raw_pre_buffer)
            log.debug("[%s] _finalise: voting collar across %d raw frames "
                      "(%d pre-trigger + %d recording)",
                      self.cam_label, len(_all_raw),
                      len(ev.trigger_raw_frames), len(self._raw_pre_buffer))

            mount_dets  = ev.trigger_dets
            has_mounter = "mounter" in mount_dets
            has_mountee = "mountee" in mount_dets

            
            if not has_mountee and ev.best_mountee_det is not None:
                mount_dets = dict(mount_dets)
                mount_dets["mountee"] = [ev.best_mountee_det]
                has_mountee = True

            
            
            if not has_mountee and ev.best_mountee_det is not None:
                mount_dets = dict(mount_dets)   
                mount_dets["mountee"] = [ev.best_mountee_det]
                has_mountee = True
            
            ref_raw = _all_raw[-1] if _all_raw else None

            if has_mounter and ref_raw is not None:
                mounter_id, mounter_id_conf, mounter_id_method, mounter_collar_xyxy = \
                    vote_collar(ref_raw, mount_dets["mounter"],
                                self.collar_model,
                                window_frames=_all_raw)
            else:
                mounter_id, mounter_id_conf, mounter_id_method, mounter_collar_xyxy = \
                    "N/A", "N/A", "not_detected", None

            if has_mountee and ref_raw is not None:
                mountee_id, mountee_id_conf, mountee_id_method, mountee_collar_xyxy = \
                    vote_collar(ref_raw, mount_dets["mountee"],
                                self.collar_model,
                                window_frames=_all_raw)
            else:
                mountee_id, mountee_id_conf, mountee_id_method, mountee_collar_xyxy = \
                    "N/A", "N/A", "not_detected", None

            
            mounter_det = max(mount_dets["mounter"], key=lambda d: d["conf"]) if has_mounter else None
            mountee_det = max(mount_dets["mountee"], key=lambda d: d["conf"]) if has_mountee else None
            if has_mounter and has_mountee and ref_raw is not None and (
                mounter_id_method == "collar" and mountee_id_method == "collar"
                and mounter_id != "N/A" and mountee_id != "N/A"
                and mounter_id.lower() == mountee_id.lower()
            ):
                disputed = mounter_id
                mr_hits = sum(1 for d in mount_dets["mounter"]
                              if _quick_collar_label(ref_raw, d["xyxy"], self.collar_model)
                                 == disputed.lower())
                me_hits = sum(1 for d in mount_dets["mountee"]
                              if _quick_collar_label(ref_raw, d["xyxy"], self.collar_model)
                                 == disputed.lower())
                log.warning("[%s] Same-collar conflict (%s) mounter:%d mountee:%d",
                            self.cam_label, disputed, mr_hits, me_hits)
                if mr_hits == me_hits:
                    coat, cc, _, _ = _coat_colour_only(ref_raw, mounter_det["xyxy"], self.collar_model)
                    mounter_id, mounter_id_conf, mounter_id_method, mounter_collar_xyxy = \
                        f"{disputed}({coat})", cc, "collar+coat_colour", None
                    coat, cc, _, _ = _coat_colour_only(ref_raw, mountee_det["xyxy"], self.collar_model)
                    mountee_id, mountee_id_conf, mountee_id_method, mountee_collar_xyxy = \
                        f"{disputed}({coat})", cc, "collar+coat_colour", None
                elif mr_hits > me_hits:
                    mountee_id, mountee_id_conf, mountee_id_method, mountee_collar_xyxy = \
                        _coat_colour_only(ref_raw, mountee_det["xyxy"], self.collar_model)
                else:
                    mounter_id, mounter_id_conf, mounter_id_method, mounter_collar_xyxy = \
                        _coat_colour_only(ref_raw, mounter_det["xyxy"], self.collar_model)

            log.info("[%s] Collar ID — mounter: %s (%s)  mountee: %s (%s)",
                     self.cam_label, mounter_id, mounter_id_method,
                     mountee_id, mountee_id_method)

            
            ev.mounter_id        = mounter_id
            ev.mounter_id_conf   = mounter_id_conf
            ev.mounter_id_method = mounter_id_method
            ev.mountee_id        = mountee_id
            ev.mountee_id_conf   = mountee_id_conf
            ev.mountee_id_method = mountee_id_method

            
            if ev.mountee_conf == 0.0 and self._peak_mountee_conf > 0.0:
                ev.mountee_conf = self._peak_mountee_conf
            self._peak_mountee_conf = 0.0

            
            if ev.mountee_conf == 0.0 and self._peak_mountee_conf > 0.0:
                ev.mountee_conf = self._peak_mountee_conf
            self._peak_mountee_conf = 0.0  

            
            write_clip(ev.frames, ev.fps, ev.clip_path)
            self.events.append(ev)
            log.info("[%s] Clip written → %s  (%d frames)",
                     self.cam_label, ev.clip_path.name, len(ev.frames))
            self._raw_pre_buffer.clear()  

    def flush(self) -> None:
        """Called at EOF — finalise any in-progress recording."""
        if self._recording and self._current_event is not None:
            ev = self._current_event
            self._current_event = None
            self._recording     = False
            log.info("[%s] EOF mid-recording — flushing partial clip (%d frames).",
                     self.cam_label, len(ev.frames))
            write_clip(ev.frames, ev.fps, ev.clip_path)
            self.events.append(ev)

    def release(self) -> None:
        self.cap.release()






_TS_PATTERNS = [
    
    (re.compile(r"(\d{4})[_\-](\d{2})[_\-](\d{2})[T _\-](\d{2})[:\-](\d{2})[:\-](\d{2})"),
     "%Y %m %d %H %M %S"),
    
    (re.compile(r"(\d{4})(\d{2})(\d{2})[_\-]?(\d{2})(\d{2})(\d{2})"),
     "%Y %m %d %H %M %S"),
]


def _path_to_epoch(path: Path) -> float:
    """
    Extract a wall-clock start timestamp from a video file.
    Priority:
      1. Timestamp embedded in the filename (various formats).
      2. File modification time (mtime) as a rough proxy.
    """
    stem = path.stem
    for pattern, fmt in _TS_PATTERNS:
        m = pattern.search(stem)
        if m:
            ts_str = " ".join(m.groups())
            try:
                dt = datetime.strptime(ts_str, fmt)
                return dt.replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                pass
    
    return path.stat().st_mtime





def resolve_input(path: Path) -> list[tuple[Path, float]]:
    """
    Accept either a single video file or a folder of video files.

    • File  → returns [(path, epoch)]  immediately.
    • Folder → scans for recognised video extensions, sorted lexicographically
               so cam1_001.mp4 < cam1_002.mp4 etc.

    Returns a list of (video_path, start_epoch) tuples.
    """
    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(
                f"{path} is not a recognised video file "
                f"(expected one of {sorted(VIDEO_EXTENSIONS)})"
            )
        epoch = _path_to_epoch(path)
        log.info("  [single file] %s  start≈%s", path.name,
                 datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S"))
        return [(path, epoch)]

    if path.is_dir():
        videos = sorted(
            p for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not videos:
            log.warning("No video files found in %s", path)
        result = [(p, _path_to_epoch(p)) for p in videos]
        for p, epoch in result:
            log.info("  %s  start≈%s", p.name,
                     datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S"))
        return result

    raise FileNotFoundError(f"Path does not exist or is not a file/folder: {path}")



scan_folder = resolve_input





def process_video_pair(
    path1: Path, epoch1: float,
    path2: Path, epoch2: float,
    mount_model: YOLO,
    collar_model: YOLO,
) -> tuple[list[MountEvent], list[MountEvent]]:
    """
    Step cam1 clip[i] and cam2 clip[i] frame-by-frame simultaneously in a
    single loop.  Both VideoProcessors share wall-clock time derived from their
    respective start epochs, so event timestamps are directly comparable.

    Returns (cam1_events, cam2_events) — individual clips already written to
    disk inside _finalise(); caller only needs the event lists for fusion.
    """
    proc1 = VideoProcessor(str(path1), "CAM1", ROI_1,
                           mount_model, collar_model, epoch1)
    proc2 = VideoProcessor(str(path2), "CAM2", ROI_2,
                           mount_model, collar_model, epoch2)

    REPORT_EVERY = 500
    frame_count  = 0

    while True:
        frame1, ts1, wepoch1 = proc1.read_frame()
        frame2, ts2, wepoch2 = proc2.read_frame()

        if frame1 is None and frame2 is None:
            break

        if frame1 is not None:
            proc1.process_frame(frame1, ts1, wepoch1)
        if frame2 is not None:
            proc2.process_frame(frame2, ts2, wepoch2)

        frame_count += 1
        if frame_count % REPORT_EVERY == 0:
            pct1 = (f"{100 * proc1.frame_idx / proc1.total_frames:.1f}%"
                    if proc1.total_frames else f"{proc1.frame_idx}f")
            pct2 = (f"{100 * proc2.frame_idx / proc2.total_frames:.1f}%"
                    if proc2.total_frames else f"{proc2.frame_idx}f")
            log.info("Progress  CAM1=%s  CAM2=%s", pct1, pct2)

    proc1.flush()
    proc2.flush()
    proc1.release()
    proc2.release()

    def _fps_summary(proc: VideoProcessor) -> str:
        times = proc._inference_times
        if not times:
            return "N/A"
        avg = sum(times) / len(times)
        return f"{1/avg:.1f} fps (avg)  {1/max(times):.1f} fps (min)  {len(times)} frames"
    log.info("Pair done — CAM1: %d event(s)  CAM2: %d event(s)",
             len(proc1.events), len(proc2.events))
    log.info("Inference speed  CAM1: %s", _fps_summary(proc1))
    log.info("Inference speed  CAM2: %s", _fps_summary(proc2))
    return proc1.events, proc2.events


def process_folder_solo(
    videos: list[tuple[Path, float]],
    cam_label: str,
    roi: tuple,
    mount_model: YOLO,
    collar_model: YOLO,
) -> list[MountEvent]:
    """
    Process remaining (unmatched) clips from one camera sequentially.
    Used when one folder has more videos than the other.
    """
    all_events: list[MountEvent] = []
    REPORT_EVERY = 500

    for video_path, start_epoch in videos:
        log.info("=== [%s solo] Processing %s ===", cam_label, video_path.name)
        proc = VideoProcessor(str(video_path), cam_label, roi,
                              mount_model, collar_model, start_epoch)
        frame_count = 0
        while True:
            frame, ts_str, wall_epoch = proc.read_frame()
            if frame is None:
                break
            proc.process_frame(frame, ts_str, wall_epoch)
            frame_count += 1
            if frame_count % REPORT_EVERY == 0:
                pct = (f"{100 * frame_count / proc.total_frames:.1f}%"
                       if proc.total_frames else f"{frame_count} frames")
                log.info("[%s] %s — %s", cam_label, video_path.name, pct)
        proc.flush()
        _times = proc._inference_times
        if _times:
            _avg = sum(_times) / len(_times)
            log.info("[%s] %s — inference: %.1f fps avg  %.1f fps min  (%d frames)",
                     cam_label, video_path.name,
                     1/_avg, 1/max(_times), len(_times))
        proc.release()
        all_events.extend(proc.events)

    return all_events





def _best_id(
    id_a: str, conf_a: float | str, method_a: str,
    id_b: str, conf_b: float | str, method_b: str,
) -> tuple[str, float | str, str]:
    """
    Choose the better identification from two cameras for one animal role.

    Priority:
      1. collar > coat_colour (method rank)
      2. Within same method: higher numeric confidence wins.
      3. If one is N/A, take the other.
    """
    METHOD_RANK = {"collar": 3, "collar+coat_colour": 2, "coat_colour": 1, "N/A": 0, "": 0,
                   "not_detected": 0}
    rank_a = METHOD_RANK.get(method_a, 0)
    rank_b = METHOD_RANK.get(method_b, 0)

    if id_a in ("N/A", "") and id_b not in ("N/A", ""):
        return id_b, conf_b, method_b
    if id_b in ("N/A", "") and id_a not in ("N/A", ""):
        return id_a, conf_a, method_a

    if rank_a > rank_b:
        return id_a, conf_a, method_a
    if rank_b > rank_a:
        return id_b, conf_b, method_b

    
    fa = float(conf_a) if isinstance(conf_a, float) else 0.0
    fb = float(conf_b) if isinstance(conf_b, float) else 0.0
    return (id_a, conf_a, method_a) if fa >= fb else (id_b, conf_b, method_b)


def _consensus_id(id1: str, conf1, id2: str, conf2) -> str:
    """Cross-camera mounter/mountee best guess.

    Rules:
    - Both agree → return that ID.
    - One is N/A or empty → return the other.
    - Disagree → return the higher-confidence one, flagged with a '?' suffix
      so it's visible in the CSV that there was a disagreement.
    """
    na = {"N/A", "", "not_detected"}
    id1_valid = id1 not in na
    id2_valid = id2 not in na

    if not id1_valid and not id2_valid:
        return "N/A"
    if not id1_valid:
        return id2
    if not id2_valid:
        return id1
    if id1 == id2:
        return id1
    
    f1 = float(conf1) if isinstance(conf1, (int, float)) else 0.0
    f2 = float(conf2) if isinstance(conf2, (int, float)) else 0.0
    winner = id1 if f1 >= f2 else id2
    return f"{winner}?"


def fuse_events(
    cam1_events: list[MountEvent],
    cam2_events: list[MountEvent],
    fusion_window_sec: float = FUSION_WINDOW_SEC,
) -> list[MountEvent]:
    """
    Greedy O(n*m) fusion of events within fusion_window_sec of each other.

    Fusion rules
    ────────────
    • event_type  — confirmed beats possible (voting).
    • confidences — max() per class.
    • IDs         — _best_id(): collar > coat_colour, then higher conf.
    • Fused events — recorded in CSV only (no fused clip written).
                     Individual camera clips remain in confirmed/possible.
                     Both-possible pairs are NOT fused; each stays as an
                     independent entry in clips/possible/.
    """
    cam1_sorted = sorted(cam1_events, key=lambda e: e.wall_epoch)
    cam2_sorted = sorted(cam2_events, key=lambda e: e.wall_epoch)
    used2       = [False] * len(cam2_sorted)
    fused_out:  list[MountEvent] = []
    unmatched1: list[MountEvent] = []

    for ev1 in cam1_sorted:
        best_j   = -1
        best_gap = float("inf")
        for j, ev2 in enumerate(cam2_sorted):
            if used2[j]:
                continue
            gap = abs(ev1.wall_epoch - ev2.wall_epoch)
            if gap < fusion_window_sec and gap < best_gap:
                best_gap = gap
                best_j   = j

        if best_j == -1:
            unmatched1.append(ev1)
            continue

        ev2 = cam2_sorted[best_j]

        
        fused_type = (
            "confirmed"
            if "confirmed" in (ev1.event_type, ev2.event_type)
            else "possible"
        )

        
        if fused_type == "possible":
            log.info(
                "SKIP FUSION  CAM1@%s [possible] ↔ CAM2@%s [possible]  gap=%.1fs — "
                "both possible, keeping as separate clips.",
                ev1.timestamp, ev2.timestamp, best_gap,
            )
            unmatched1.append(ev1)
            used2[best_j] = True   
            continue

        used2[best_j] = True
        log.info("FUSING  CAM1@%s [%s]  ↔  CAM2@%s [%s]  (gap=%.1fs)",
                 ev1.timestamp, ev1.event_type,
                 ev2.timestamp, ev2.event_type, best_gap)

        
        fused_me_conf  = max(ev1.mount_event_conf, ev2.mount_event_conf)
        fused_mr_conf  = max(ev1.mounter_conf,     ev2.mounter_conf)
        fused_me2_conf = max(ev1.mountee_conf,     ev2.mountee_conf)

        
        f_mounter_id, f_mounter_conf, f_mounter_method = _best_id(
            ev1.mounter_id, ev1.mounter_id_conf, ev1.mounter_id_method,
            ev2.mounter_id, ev2.mounter_id_conf, ev2.mounter_id_method,
        )
        f_mountee_id, f_mountee_conf, f_mountee_method = _best_id(
            ev1.mountee_id, ev1.mountee_id_conf, ev1.mountee_id_method,
            ev2.mountee_id, ev2.mountee_id_conf, ev2.mountee_id_method,
        )

        
        
        primary = ev1 if ev1.mean_det_conf >= ev2.mean_det_conf else ev2

        
        _c1, _c2 = (ev1, ev2) if "CAM1" in ev1.cam else (ev2, ev1)
        fused_ev = MountEvent(
            cam               = f"{ev1.cam}+{ev2.cam}",
            video_path        = f"{ev1.video_path} | {ev2.video_path}",
            timestamp         = primary.timestamp,
            wall_epoch        = primary.wall_epoch,
            event_type        = "confirmed",
            mount_event_conf  = fused_me_conf,
            mounter_conf      = fused_mr_conf,
            mountee_conf      = fused_me2_conf,
            mounter_id        = f_mounter_id,
            mounter_id_conf   = f_mounter_conf,
            mounter_id_method = f_mounter_method,
            mountee_id        = f_mountee_id,
            mountee_id_conf   = f_mountee_conf,
            mountee_id_method = f_mountee_method,
            clip_name         = f"{ev1.clip_name} | {ev2.clip_name}",
            clip_path         = ev1.clip_path,
            fps               = primary.fps,
            fused             = True,
            fused_with_clip   = ev2.clip_name if primary is ev1 else ev1.clip_name,
            cam1_mount_event_conf = _c1.mount_event_conf,
            cam1_mounter_conf     = _c1.mounter_conf,
            cam1_mountee_conf     = _c1.mountee_conf,
            cam2_mount_event_conf = _c2.mount_event_conf,
            cam2_mounter_conf     = _c2.mounter_conf,
            cam2_mountee_conf     = _c2.mountee_conf,
            
            cam1_mounter_id      = _c1.mounter_id,
            cam1_mounter_id_conf = _c1.mounter_id_conf,
            cam1_mountee_id      = _c1.mountee_id,
            cam1_mountee_id_conf = _c1.mountee_id_conf,
            cam2_mounter_id      = _c2.mounter_id,
            cam2_mounter_id_conf = _c2.mounter_id_conf,
            cam2_mountee_id      = _c2.mountee_id,
            cam2_mountee_id_conf = _c2.mountee_id_conf,
            
            
            possible_mounter     = _consensus_id(_c1.mounter_id, _c1.mounter_id_conf,
                                                  _c2.mounter_id, _c2.mounter_id_conf),
            possible_mountee     = _consensus_id(_c1.mountee_id, _c1.mountee_id_conf,
                                                  _c2.mountee_id, _c2.mountee_id_conf),
        )
        fused_out.append(fused_ev)

        log.info(
            "  Fused confirmed → mounter=%s (%s) | mountee=%s (%s)",
            f_mounter_id, f_mounter_method,
            f_mountee_id, f_mountee_method,
        )

    
    unmatched2 = [ev2 for j, ev2 in enumerate(cam2_sorted) if not used2[j]]
    for ev in unmatched1 + unmatched2:
        log.info("Unmatched [%s] event @ %s — clip already on disk: %s",
                 ev.cam, ev.timestamp, ev.clip_name)

    all_out = sorted(fused_out + unmatched1 + unmatched2, key=lambda e: e.wall_epoch)
    log.info("Fusion complete: %d fused, %d CAM1-only, %d CAM2-only → %d total events.",
             len(fused_out), len(unmatched1), len(unmatched2), len(all_out))
    return all_out






def process_rtsp(
    cam1_url: str,
    cam2_url: str | None,
    mount_model: YOLO,
    collar_model: YOLO,
    fusion_window_sec: float = FUSION_WINDOW_SEC,
) -> None:
    """
    Run the detector continuously on one or two live RTSP streams.
    Reconnects automatically if the stream drops.
    Press Ctrl+C to stop.
    """
    import signal

    stop_flag = {"stop": False}

    def _sigint(sig, frame):
        log.info("Interrupted — stopping after current frame.")
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _sigint)

    RECONNECT_DELAY = 5  

    def _make_proc(url: str, cam_label: str, roi: tuple) -> VideoProcessor:
        while not stop_flag["stop"]:
            proc = VideoProcessor(url, cam_label, roi, mount_model, collar_model,
                                  video_start_epoch=time.time())
            if proc.cap.isOpened():
                log.info("[%s] Connected to %s", cam_label, url)
                return proc
            log.warning("[%s] Could not connect to %s — retrying in %ds",
                        cam_label, url, RECONNECT_DELAY)
            time.sleep(RECONNECT_DELAY)
        return None

    log.info("=== RTSP Live Mode ===")
    log.info("CAM1: %s", cam1_url)
    if cam2_url:
        log.info("CAM2: %s", cam2_url)
    log.info("Press Ctrl+C to stop.")

    fused_events: list[MountEvent] = []
    frame_count  = 0
    REPORT_EVERY = 300

    proc1 = _make_proc(cam1_url, "CAM1", ROI_1)
    proc2 = _make_proc(cam2_url, "CAM2", ROI_2) if cam2_url else None

    while not stop_flag["stop"]:
        
        frame1, ts1, wepoch1 = proc1.read_frame()
        if frame1 is None:
            log.warning("[CAM1] Stream lost — reconnecting...")
            proc1.flush()
            proc1.release()
            proc1 = _make_proc(cam1_url, "CAM1", ROI_1)
            if proc1 is None:
                break
            continue
        proc1.process_frame(frame1, ts1, wepoch1)

        
        if proc2 is not None:
            frame2, ts2, wepoch2 = proc2.read_frame()
            if frame2 is None:
                log.warning("[CAM2] Stream lost — reconnecting...")
                proc2.flush()
                proc2.release()
                proc2 = _make_proc(cam2_url, "CAM2", ROI_2)
                if proc2 is None:
                    break
                continue
            proc2.process_frame(frame2, ts2, wepoch2)

        frame_count += 1
        if frame_count % REPORT_EVERY == 0:
            e1 = len(proc1.events)
            e2 = len(proc2.events) if proc2 else 0
            log.info("Live: %d frames processed  CAM1=%d events  CAM2=%d events",
                     frame_count, e1, e2)

    
    log.info("Stopping — flushing buffers...")
    proc1.flush()
    proc1.release()
    if proc2:
        proc2.flush()
        proc2.release()

    if proc2:
        
        all_events = fuse_events(proc1.events, proc2.events, fusion_window_sec)
    else:
        all_events = proc1.events

    write_csv(all_events)
    log.info("Done. %d total events logged to %s", len(all_events), LOG_FILE)

def process_dual_folders(
    cam1_input: Path,
    cam2_input: Path | None,
    mount_model: YOLO,
    collar_model: YOLO,
    fusion_window_sec: float = FUSION_WINDOW_SEC,
    cam1_start_override: float | None = None,
    cam2_start_override: float | None = None,
) -> None:
    cam1_videos = resolve_input(cam1_input)
    if cam1_start_override is not None:
        cam1_videos = [(p, cam1_start_override) for p, _ in cam1_videos]
        log.info("CAM1 start overridden → %s",
                 __import__("datetime").datetime.fromtimestamp(cam1_start_override))

    if cam2_input is None:
        
        log.info("=== Single-camera mode: %s ===", cam1_input)
        all_events = process_folder_solo(
            cam1_videos, "CAM1", ROI_1, mount_model, collar_model
        )
        write_csv(all_events)
        log.info("All done.  Clips → %s | Log → %s", CLIPS_DIR, LOG_FILE)
        return

    cam2_videos = resolve_input(cam2_input)
    if cam2_start_override is not None:
        cam2_videos = [(p, cam2_start_override) for p, _ in cam2_videos]
        log.info("CAM2 start overridden → %s",
                 __import__("datetime").datetime.fromtimestamp(cam2_start_override))
    n_pairs     = min(len(cam1_videos), len(cam2_videos))

    log.info("=== Paired simultaneous processing — %d clip pair(s) ===", n_pairs)

    all_cam1_events: list[MountEvent] = []
    all_cam2_events: list[MountEvent] = []

    
    for i in range(n_pairs):
        path1, epoch1 = cam1_videos[i]
        path2, epoch2 = cam2_videos[i]
        log.info("--- Pair %d/%d: %s  ↔  %s ---", i + 1, n_pairs, path1.name, path2.name)
        ev1, ev2 = process_video_pair(
            path1, epoch1, path2, epoch2,
            mount_model, collar_model,
        )
        all_cam1_events.extend(ev1)
        all_cam2_events.extend(ev2)

        
        
        if ev1 or ev2:
            pair_fused = fuse_events(ev1, ev2, fusion_window_sec)
            
            
            
            if i == 0:
                merged_events = pair_fused
            else:
                merged_events = sorted(
                    merged_events + pair_fused, key=lambda e: e.wall_epoch
                )
        
        for ev in ev1 + ev2:
            ev.frames = []

    
    leftover_cam1 = cam1_videos[n_pairs:]
    leftover_cam2 = cam2_videos[n_pairs:]

    if leftover_cam1:
        log.info("=== Processing %d leftover CAM1 clip(s) solo ===", len(leftover_cam1))
        solo_ev = process_folder_solo(
            leftover_cam1, "CAM1", ROI_1, mount_model, collar_model
        )
        merged_events = sorted(merged_events + solo_ev, key=lambda e: e.wall_epoch)

    if leftover_cam2:
        log.info("=== Processing %d leftover CAM2 clip(s) solo ===", len(leftover_cam2))
        solo_ev = process_folder_solo(
            leftover_cam2, "CAM2", ROI_2, mount_model, collar_model
        )
        merged_events = sorted(merged_events + solo_ev, key=lambda e: e.wall_epoch)

    if not (all_cam1_events or all_cam2_events):
        merged_events = []

    write_csv(merged_events)
    log.info(
        "All done — %d total events (%d fused confirmed, %d individual).  "
        "Clips → %s | Log → %s",
        len(merged_events),
        sum(1 for e in merged_events if e.fused),
        sum(1 for e in merged_events if not e.fused),
        CLIPS_DIR, LOG_FILE,
    )





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cattle Mount Detection — Dual-Folder Mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cam1", required=True, type=Path,
        help="CAM1 input: a single video file OR a folder of video files",
    )
    parser.add_argument(
        "--cam2", default=None, type=Path,
        help="CAM2 input: a single video file OR a folder of video files "
             "(omit for single-camera mode)",
    )
    parser.add_argument("--live", action="store_true",
                        help="Live RTSP mode")
    parser.add_argument(
        "--mount-model", default=MOUNT_MODEL_PATH,
        help="Path to mount detection model (.pt)",
    )
    parser.add_argument(
        "--collar-model", default=COLLAR_MODEL_PATH,
        help="Path to collar identification model (.pt)",
    )
    parser.add_argument(
        "--mount-conf", type=float, default=MOUNT_CONF,
        help="Global confidence threshold for mount detection",
    )
    parser.add_argument(
        "--collar-conf", type=float, default=COLLAR_CONF,
        help="Confidence threshold for collar detection",
    )
    parser.add_argument(
        "--cam1-start", default=None, type=float,
        help="Override CAM1 start epoch (Unix timestamp). "
             "Use to fix wrong file timestamps for fusion.",
    )
    parser.add_argument(
        "--cam2-start", default=None, type=float,
        help="Override CAM2 start epoch (Unix timestamp). "
             "Use to fix wrong file timestamps for fusion.",
    )
    parser.add_argument(
        "--fusion-window", type=float, default=FUSION_WINDOW_SEC,
        help="Max time gap (seconds) between two cross-camera events to be fused",
    )
    parser.add_argument(
        "--mount-augment", action="store_true",
        help="Enable YOLO test-time augmentation for the mount model.  "
             "Costs ~2× inference time but can recover small/flickery classes.",
    )
    parser.add_argument(
        "--mount-imgsz", type=int, default=MOUNT_IMGSZ,
        help="Inference image size for the mount model (larger → better recall, slower).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global MOUNT_CONF, COLLAR_CONF
    global MOUNT_MODEL_PATH, COLLAR_MODEL_PATH
    global MOUNT_AUGMENT, MOUNT_IMGSZ
    MOUNT_CONF        = args.mount_conf
    COLLAR_CONF       = args.collar_conf
    MOUNT_MODEL_PATH  = args.mount_model
    COLLAR_MODEL_PATH = args.collar_model
    MOUNT_AUGMENT     = args.mount_augment
    MOUNT_IMGSZ       = args.mount_imgsz

    log.info("=== Cattle Mount Detection (Dual-Folder Mode) ===")
    log.info("Mount model              : %s  (conf=%.2f)", MOUNT_MODEL_PATH, MOUNT_CONF)
    log.info("Collar model             : %s  (conf=%.2f)", COLLAR_MODEL_PATH, COLLAR_CONF)
    log.info("CAM1 input               : %s", args.cam1)
    log.info("CAM2 input               : %s", args.cam2 or "disabled")
    log.info("Fusion window            : %.1fs", args.fusion_window)
    log.info("Confirm frames (n)       : %d  (%.1fs @ %.0f fps)",
             round(MOUNT_CONFIRM_SEC * 15), MOUNT_CONFIRM_SEC, 15.0)

    mount_model  = YOLO(MOUNT_MODEL_PATH)
    collar_model = YOLO(COLLAR_MODEL_PATH)
    mount_model.fuse()
    collar_model.fuse()
    if USE_HALF:
        mount_model.model.half()
        collar_model.model.half()
    log.info("Inference device         : %s%s", DEVICE,
             "  (FP16)" if USE_HALF else "")
    log.info("Mount imgsz              : %d", MOUNT_IMGSZ)
    log.info("Mount test-time augment  : %s", "on" if MOUNT_AUGMENT else "off")

    if args.live:
        process_rtsp(
            cam1_url          = args.cam1,
            cam2_url          = args.cam2,
            mount_model       = mount_model,
            collar_model      = collar_model,
            fusion_window_sec = args.fusion_window,
        )
    else:
        process_dual_folders(
            cam1_input           = args.cam1,
            cam2_input           = args.cam2,
            mount_model          = mount_model,
            collar_model         = collar_model,
            fusion_window_sec    = args.fusion_window,
            cam1_start_override  = args.cam1_start,
            cam2_start_override  = args.cam2_start,
        )


if __name__ == "__main__":
    main()
