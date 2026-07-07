"""
CowEstrus Dashboard Backend
============================
Serves the dashboard HTML and exposes API endpoints that read the
mount_log.csv and manage the detection subprocess.

Usage:
    pip install flask opencv-python --break-system-packages
    python3 dashboard_server.py

Then open http://<jetson-ip>:5000 in your browser.
"""

import csv
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, request, send_from_directory

app = Flask(__name__, static_folder=".")

LOG_FILE  = Path("logs/mount_log.csv")
CLIPS_DIR = Path("clips")

# ── Subprocess state ──────────────────────────────────────────────────────────
_proc       = None
_proc_lock  = threading.Lock()
_stop_event = threading.Event()   # signals capture threads to quit

# ── FPS tracking ─────────────────────────────────────────────────────────────
_fps_state = {"cam1": None, "cam2": None}
_fps_lock  = threading.Lock()
FPS_PATTERN = re.compile(r'\[(?P<cam>CAM[12])\].*?(?P<fps>[\d.]+)\s*fps', re.IGNORECASE)

# ── Frame buffers ─────────────────────────────────────────────────────────────
_frames      = {"cam1": None, "cam2": None}
_frames_lock = threading.Lock()


# ── Background workers ────────────────────────────────────────────────────────

def _tail_output(proc):
    """Read stdout/stderr, print to terminal, and extract FPS; detect when process finishes."""
    global _proc
    for line in proc.stdout:
        line = line.strip()
        print(f"[DETECTOR] {line}", flush=True)   # ← show all detector output in terminal
        m = FPS_PATTERN.search(line)
        if m:
            with _fps_lock:
                key = m.group("cam").lower()
                _fps_state[key] = m.group("fps")
    # Process has ended — update state
    _stop_event.set()
    with _proc_lock:
        _proc = None
    print("[DETECTOR] Process ended.", flush=True)


def _capture_frames(cam_key: str, source: str) -> None:
    """Read frames from video/RTSP and store the latest JPEG."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[stream] Could not open {source}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_video     = total_frames > 0   # False for RTSP

    while not _stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            if is_video:
                # Loop video for preview
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break  # RTSP disconnected

        # Resize for bandwidth
        h, w = frame.shape[:2]
        scale = min(1.0, 640 / max(w, 1))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        with _frames_lock:
            _frames[cam_key] = jpeg.tobytes()

        time.sleep(1 / 15)  # ~15 fps preview

    cap.release()
    with _frames_lock:
        _frames[cam_key] = None


def _watch_proc():
    """Poll subprocess; set stop event when it finishes naturally."""
    global _proc
    while True:
        time.sleep(1)
        with _proc_lock:
            p = _proc
        if p is None:
            break
        if p.poll() is not None:
            _stop_event.set()
            with _proc_lock:
                _proc = None
            break


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "dashboard.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    global _proc
    data = request.get_json(force=True)
    cmd  = data.get("command", "").strip()
    if not cmd:
        return jsonify(ok=False, error="No command provided"), 400

    with _proc_lock:
        if _proc and _proc.poll() is None:
            return jsonify(ok=False, error="Already running"), 409
        try:
            print(f"\n[SERVER] Launching command:\n  {cmd}\n", flush=True)
            _stop_event.clear()
            with _frames_lock:
                _frames["cam1"] = None
                _frames["cam2"] = None

            kwargs = dict(
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if hasattr(os, "setsid"):
                kwargs["preexec_fn"] = os.setsid

            _proc = subprocess.Popen(cmd, **kwargs)

            # Tail stdout for FPS
            threading.Thread(target=_tail_output, args=(_proc,), daemon=True).start()
            # Watch for natural exit
            threading.Thread(target=_watch_proc, daemon=True).start()

            # Frame capture threads
            cam1_src = data.get("cam1", "")
            cam2_src = data.get("cam2", "")
            if cam1_src:
                threading.Thread(target=_capture_frames, args=("cam1", cam1_src), daemon=True).start()
            if cam2_src:
                threading.Thread(target=_capture_frames, args=("cam2", cam2_src), daemon=True).start()

            return jsonify(ok=True, pid=_proc.pid)
        except Exception as e:
            return jsonify(ok=False, error=str(e)), 500


@app.route("/api/stop", methods=["POST"])
def api_stop():
    global _proc
    _stop_event.set()
    with _proc_lock:
        if _proc and _proc.poll() is None:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(_proc.pid), signal.SIGTERM)
                else:
                    _proc.terminate()
            except Exception:
                try:
                    _proc.terminate()
                except Exception:
                    pass
            _proc = None
    return jsonify(ok=True)


@app.route("/api/status")
def api_status():
    running = bool(_proc and _proc.poll() is None)
    with _fps_lock:
        fps = dict(_fps_state)
    return jsonify(running=running, fps=fps, stopped=_stop_event.is_set())


@app.route("/api/events")
def api_events():
    events = []
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    events.append(dict(row))
        except Exception as e:
            return jsonify(events=[], error=str(e))

    with _fps_lock:
        fps = dict(_fps_state)

    running = bool(_proc and _proc.poll() is None)
    return jsonify(events=events, fps=fps, running=running,
                   stopped=_stop_event.is_set(), total=len(events))


@app.route("/api/clips")
def api_clips():
    clips = []
    for subdir in ("confirmed", "possible", "fused"):
        d = CLIPS_DIR / subdir
        if d.exists():
            for f in sorted(d.iterdir()):
                if f.suffix == ".mp4":
                    clips.append({"name": f.name, "type": subdir,
                                  "path": str(f),
                                  "size_mb": round(f.stat().st_size / 1e6, 1)})
    return jsonify(clips=clips)


@app.route("/clips/<path:filename>")
def serve_clip(filename):
    """Search all subdirs for the clip and stream it."""
    from flask import send_file
    search_dirs = []
    if CLIPS_DIR.exists():
        search_dirs.append(CLIPS_DIR)
        search_dirs += [d for d in CLIPS_DIR.iterdir() if d.is_dir()]
    for d in search_dirs:
        candidate = d / filename
        if candidate.exists():
            return send_file(
                str(candidate.resolve()),
                mimetype="video/mp4",
                conditional=True,   # supports Range requests for seek
                as_attachment=False,
            )
    return jsonify(error="Clip not found: " + filename), 404


# ── MJPEG streams ─────────────────────────────────────────────────────────────

PLACEHOLDER_JPEG = None

def _get_placeholder():
    """1x1 grey JPEG for when no frame is available."""
    global PLACEHOLDER_JPEG
    if PLACEHOLDER_JPEG is None:
        import numpy as np
        img = np.full((360, 640, 3), 30, dtype='uint8')
        _, buf = cv2.imencode('.jpg', img)
        PLACEHOLDER_JPEG = buf.tobytes()
    return PLACEHOLDER_JPEG


def _mjpeg_gen(cam_key: str):
    while True:
        with _frames_lock:
            frame = _frames.get(cam_key)
        data = frame if frame else _get_placeholder()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        time.sleep(1 / 15)


@app.route("/stream/cam1")
def stream_cam1():
    return Response(_mjpeg_gen("cam1"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stream/cam2")
def stream_cam2():
    return Response(_mjpeg_gen("cam2"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")



@app.route("/api/review", methods=["POST"])
def api_review():
    """Copy a confirmed clip into the farmer_confirmed folder."""
    import shutil
    data     = request.get_json(force=True)
    clip     = data.get("clip", "").strip()
    decision = data.get("decision", "")

    if not clip or decision != "farmer_confirmed":
        return jsonify(ok=False, error="Invalid request"), 400

    # Search for the clip across all subdirs
    src_path = None
    for subdir in ("", "confirmed", "possible", "fused"):
        candidate = CLIPS_DIR / subdir / clip if subdir else CLIPS_DIR / clip
        if candidate.exists():
            src_path = candidate
            break

    if not src_path:
        return jsonify(ok=False, error="Clip not found: " + clip), 404

    dest_dir = CLIPS_DIR / "farmer_confirmed"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / clip

    try:
        shutil.copy2(str(src_path), str(dest_path))
        return jsonify(ok=True, dest=str(dest_path))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


@app.route("/api/open_clip", methods=["POST"])
def api_open_clip():
    """Open a clip in the system default video player on the Jetson."""
    import shutil
    data     = request.get_json(force=True)
    filename = data.get("clip", "").strip()
    if not filename:
        return jsonify(ok=False, error="No clip specified"), 400

    # Fused events store "clip1.mp4 | clip2.mp4" — take the first one
    filename = filename.split("|")[0].strip()

    def find_clip(name):
        if not CLIPS_DIR.exists():
            return None
        for d in [CLIPS_DIR] + sorted([d for d in CLIPS_DIR.iterdir() if d.is_dir()]):
            candidate = d / name
            if candidate.exists():
                return candidate
        return None

    clip_path = find_clip(filename)
    if not clip_path:
        return jsonify(ok=False, error="Clip not found: " + filename), 404

    # Try players in order of preference
    players = ["vlc", "mpv", "totem", "ffplay", "xdg-open"]
    player  = None
    for p in players:
        if shutil.which(p):
            player = p
            break

    if not player:
        return jsonify(ok=False, error="No video player found. Run: sudo apt install vlc"), 404

    try:
        subprocess.Popen([player, str(clip_path.resolve())],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
        return jsonify(ok=True, player=player, clip=str(clip_path))
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 500


# ── Health ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify(ok=True)


if __name__ == "__main__":
    print("=" * 50)
    print("  CowEstrus Dashboard  —  http://0.0.0.0:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
