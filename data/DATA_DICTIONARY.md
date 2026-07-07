# Data Dictionary — Evaluation Records

This folder contains the per-event detection logs used to compute every quantitative
result reported in the manuscript. Each row is **one logged detection event** (one saved
clip), not necessarily one distinct ground-truth mounting event. Recall metrics are scored
by mapping these logged detections onto the set of known ground-truth events defined in the
experimental protocol (see *Provenance* below); the raw row count of a file is therefore not
the same as a metric numerator.

All confidence-interval and significance calculations consume **counts** derived from these
files and are reproduced by `../compute_cis.py`.

---

## Files

| File | Experiment | Rows | What each row is | Primary metrics it supports |
|------|-----------|------|------------------|-----------------------------|
| `CONTROLLED_RECALL_TEST.csv` | Controlled recall test — staged mounting events with known ground truth | 17 | A detection the system logged for a staged event | Event-level recall, per-camera recall, confirmed-tier recall (TP/FN), single- vs. fused McNemar |
| `NIGHTTIME_FALSE_POSITIVE_TEST.csv` | Nighttime false-positive test — footage with no true mounting events | 9 | A false detection logged overnight | Confirmed-tier (nighttime) false-positive rate |
| `LONG_VIDEO_TEST.csv` | Continuous long-run test — uninterrupted multi-hour footage | 24 | A detection logged during the continuous run, hand-scored as true/false | Overall & confirmed-tier false-positive counts, mounter/mountee/overall ID accuracy |

---

## Shared column schema

`CONTROLLED_RECALL_TEST.csv` and `NIGHTTIME_FALSE_POSITIVE_TEST.csv` share the same 31-column
schema. `LONG_VIDEO_TEST.csv` uses a reduced 24-column variant (it drops the per-camera ID
columns and the `possible_*` columns, and adds three manual-scoring columns — see below).

| Column | Meaning |
|--------|---------|
| `cam` | Which stream produced the event: `CAM1`, `CAM2`, `CAM1+CAM2` / `FUSED` (cross-camera fused event). |
| `video_source` | Source clip filename(s). Fused events list both as `a.mp4 \| b.mp4`. **Filenames only** — original local paths were removed (see *Cleaning*). |
| `event_type` | Detection tier: `confirmed` or `possible`. |
| `mount_event_conf` | Detector confidence for the mount-event class on the winning stream. |
| `mounter_conf`, `mountee_conf` | Detector confidence for the mounter / mountee roles. `0` = role not detected in this event. |
| `cam1_*_conf`, `cam2_*_conf` | Per-stream confidences for the mount-event / mounter / mountee classes. Blank when that camera did not contribute. |
| `mounter_id`, `mountee_id` | Assigned individual identity (e.g. `red_collar`, `green_collar`, or a coat-colour label such as `mixed`, `grey`, `brown`, `black`). |
| `mounter_id_conf`, `mountee_id_conf` | Confidence of the identity assignment. `N/A` when identity came from coat colour rather than a collar detection. |
| `mounter_id_method`, `mountee_id_method` | How identity was assigned: `collar`, `coat_colour`, `collar+coat_colour`, or `not_detected`. |
| `cam1_mounter_id` … `cam2_mountee_id_conf` | Per-stream identity assignments and confidences (shared-schema files only). |
| `possible_mounter`, `possible_mountee` | Provisional identities recorded when an event was at the `possible` tier (shared-schema files only). A trailing `?` marks a low-confidence guess. |
| `clip_filename` | Saved clip file(s) for the event. Fused events list both. |
| `fused` | `TRUE` if this event was formed by cross-camera temporal fusion, else `FALSE`. |
| `fused_with_clip` | The partner clip a fused event was joined with (blank otherwise). |

### `LONG_VIDEO_TEST.csv` — additional manual-scoring columns

| Column | Meaning |
|--------|---------|
| `status` | Human judgement of the detection: `Positive` (a real mounting event) or `False Positive`. |
| `Mounter_ID Correct` | Whether the assigned mounter identity was right: `Correct` / `Incorrect`. Blank for false positives and for events with no assigned mounter. |
| `Mountee_ID Correct` | Same, for the mountee identity. |

> Note: in `LONG_VIDEO_TEST.csv` the `cam` column uses both `CAM1+CAM2` and `FUSED` for
> cross-camera fused events. They denote the same thing; the label is not semantically distinct.

---

## Categorical value legends

- **`event_type`**: `confirmed`, `possible`
- **`status`** (LONG_VIDEO only): `Positive`, `False Positive`
- **`*_id_method`**: `collar`, `coat_colour`, `collar+coat_colour`, `not_detected`
- **`fused`**: `TRUE`, `FALSE`
- **`Mounter_ID Correct` / `Mountee_ID Correct`**: `Correct`, `Incorrect`, or blank (not applicable)
- **Identity labels**: collar classes `blue_collar`, `green_collar`, `red_collar`, `yellow_collar`;
  coat-colour fallbacks `mixed`, `grey`, `brown`, `black`. A parenthetical (e.g. `red_collar(black)`)
  records a combined collar+coat judgement; a trailing `?` marks low confidence.

---

## Provenance — how each reported metric maps to these files

### Directly countable from the files (verified against `compute_cis.py`)

| Metric | Source file | How it is counted | Count |
|--------|-------------|-------------------|-------|
| Mounter ID accuracy | LONG_VIDEO | `Mounter_ID Correct` = `Correct` / all scored | 11 / 12 |
| Mountee ID accuracy | LONG_VIDEO | `Mountee_ID Correct` = `Correct` / all scored | 7 / 12 |
| Overall ID accuracy | LONG_VIDEO | correct mounter + correct mountee / all scored | 18 / 24 |
| Overall false positives | LONG_VIDEO | `status` = `False Positive` | 9 |
| Confirmed-tier false positives | LONG_VIDEO | `status` = `False Positive` **and** `event_type` = `confirmed` | 5 |
| Nighttime confirmed false positives | NIGHTTIME | `event_type` = `confirmed` | 1 |
| Confirmed detections (controlled) | CONTROLLED | `event_type` = `confirmed` | 12 |

### Requires the experimental protocol — NOT contained in these files (confirm before publishing)

| Quantity | Value used in `compute_cis.py` | Where it comes from |
|----------|-------------------------------|---------------------|
| Total staged ground-truth events (recall denominator) | 22 | Controlled-test protocol — the count of mounting events deliberately staged. Missed events produce **no row** in the log, so they cannot be recovered from the CSV.|
| Event-level recall numerator | 18 | Number of the 22 staged events detected at any tier, by either camera. Scored **by event identity**, not by counting log rows. **Confirm the scoring basis.** |
| Per-camera (single-stream) recall | 11 / 22 | Counterfactual: how many staged events a single camera alone would have caught. Not a stored column; derived during scoring. Tied to the McNemar test (b=7, c=0 → 11 + 7 = 18). |

---

## Data cleaning applied

The published files differ from the raw detector output in two documented ways:

1. **One duplicate row removed** from `CONTROLLED_RECALL_TEST.csv`: a redundant log line
   (`video_source = 14.mp4`) that carried the same clip filename and identical confidence
   values as the retained `12.mp4` row — a double-logged detection, not a second event.
   Removal brings the confirmed-detection count to 12, consistent with the reported TP.
2. **`video_source` reduced to bare filenames.** The raw logs stored absolute local paths
   (`C:\...\videos\...`); only the filenames are retained here. Fused pairs are preserved as
   `a.mp4 | b.mp4`.

No confidence values, identities, statuses, or scoring judgements were altered.

## Known artifacts (non-blocking)

- Some `clip_filename` / `fused_with_clip` timestamps read `1970-01-01`: the source video files
  had missing modification times, so the detector's wall-clock fell back to the Unix epoch.
  Cross-camera fusion in those runs used explicit `--cam1-start` / `--cam2-start` overrides
  rather than these timestamps. One clip carries a `2026` date from a mis-set source-file time.
  These affect filename strings only, never the scoring or the computed metrics.
