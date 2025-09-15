#!/usr/bin/env python3
# av_fuse.py  — V1.0.3
#
# What’s new vs V1.0.2:
# - ALWAYS writes outputs. If visual attribution fails, it still writes a STT-only transcript
#   (lines prefixed with "Unknown:") and diagnostics with status="stt_only".
# - Logs to console AND to a run log file: <Working>\out\<visual_basename>_run.log
# - Prints explicit file paths and stage markers. Preflight checks the visual file and first frame.
# - GUI pickers for Visual, Audio, ICS, and Working Folder (creates <Work>\out and <Work>\temp).
# - Minimal CLI (you can set CONFIG paths and run with --nogui).
#
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

# Optional deps; script degrades gracefully if missing.
try:
    import pytesseract
except Exception:
    pytesseract = None
try:
    from rapidfuzz import process, fuzz
except Exception:
    process = None
    fuzz = None
try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

# ==========================
# CONFIG (edit as needed)
# ==========================
CONFIG = {
    # Leave as None to trigger GUI pickers
    "VISUAL_VIDEO": None,  # Zoom screen-record (visual only)
    "AUDIO_SOURCE": None,  # Phone video/audio (with audio) OR audio file
    "ICS_FILE": None,      # .ics invite
    "WORK_DIR": None,      # Folder to hold out\ and temp\ (GUI prompts if None)

    # STT / model
    "WHISPER_MODEL": "medium",
    "WHISPER_DEVICE": "cuda",      # "cuda" or "cpu"
    "WHISPER_COMPUTE": "float16",  # "float16", "int8", ...

    # Visual detection
    "FPS_SAMPLE": 2.0,      # frames/sec to sample for border detection
    "OCR_ENABLED": True,    # False to disable OCR of bottom labels

    # Debug extras
    "DEBUG_SNAPSHOTS": False,   # if True, saves a few sampled frames to temp\
    "SNAPSHOT_EVERY": 150,      # snapshot every N sampled frames (if enabled)
}

USE_GUI_DEFAULT = True  # set False to rely solely on CONFIG (and run with --nogui)

# ==========================
# Logging utilities
# ==========================
_log_file_handle = None

def _open_log_file(path: Path):
    global _log_file_handle
    try:
        _log_file_handle = path.open("a", encoding="utf-8")
        _log_file_handle.write(f"===== RUN START {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        _log_file_handle.flush()
    except Exception:
        _log_file_handle = None

def _write_log_file(msg: str):
    global _log_file_handle
    if _log_file_handle:
        try:
            _log_file_handle.write(msg + "\n")
            _log_file_handle.flush()
        except Exception:
            pass

def log(msg: str):
    print(msg, flush=True)
    _write_log_file(msg)

def elog(msg: str):
    print(msg, file=sys.stderr, flush=True)
    _write_log_file("[ERR] " + msg)

def close_log():
    global _log_file_handle
    if _log_file_handle:
        try:
            _log_file_handle.write(f"===== RUN END {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            _log_file_handle.flush()
            _log_file_handle.close()
        except Exception:
            pass
        _log_file_handle = None

def run(cmd: List[str], check=True) -> subprocess.CompletedProcess:
    elog("> " + " ".join(cmd))
    return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# ==========================
# Helpers
# ==========================
def is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in {".wav", ".m4a", ".mp3", ".aac", ".flac", ".ogg", ".opus"}

def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"}

@dataclass
class Attendee:
    name: str
    email: str = ""
    company: str = ""

def parse_ics_attendees(ics_path: Path) -> List[Attendee]:
    attendees: List[Attendee] = []
    if not ics_path or not ics_path.exists():
        return attendees
    lines = ics_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    event_org = ""
    for ln in lines:
        l = ln.strip()
        if l.upper().startswith("ATTENDEE"):
            name_match = re.search(r";CN=([^;:]+)", l, flags=re.I)
            mail_match = re.search(r":mailto:([^ \r\n]+)", l, flags=re.I)
            name = name_match.group(1).strip() if name_match else ""
            email = mail_match.group(1).strip() if mail_match else ""
            name = name.replace(r"\,", ",").strip('"')
            attendees.append(Attendee(name=name, email=email, company=""))
        elif l.upper().startswith("ORGANIZATION:") or l.upper().startswith("ORG:"):
            event_org = l.split(":",1)[-1].strip()
    if event_org and attendees:
        for a in attendees:
            if not a.company:
                a.company = event_org
    return attendees

def best_icsonym_match(raw: str, icsonyms: List[str]) -> Optional[str]:
    if not raw:
        return None
    s = re.sub(r"[^A-Za-z0-9@\.\-\' ]+", "", raw).strip()
    if not s:
        return None
    if not icsonyms:
        return s
    if process and fuzz:
        m = process.extractOne(s, icsonyms, scorer=fuzz.WRatio, score_cutoff=72)
        if m:
            return m[0]
        return s
    else:
        for can in icsonyms:
            if s.lower() == can.lower():
                return can
        return s

def extract_audio_to_wav(src: Path, out_wav: Path, sr: int = 16000) -> Path:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", str(out_wav)]
    proc = run(cmd, check=True)
    if proc.returncode != 0:
        elog(proc.stderr)
        raise RuntimeError("ffmpeg failed to extract audio.")
    return out_wav

@dataclass
class STTSegment:
    start: float
    end: float
    text: str

def transcribe_wav_fwhisper(wav_path: Path, model_size: str, device: str, compute_type: str) -> List[STTSegment]:
    if WhisperModel is None:
        raise ImportError("faster-whisper is not installed. pip install faster-whisper")
    log(f"[STT] Loading Whisper model: {model_size} ({device}/{compute_type})")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments: List[STTSegment] = []
    opts = dict(beam_size=5, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
    t0 = time.time()
    it, _info = model.transcribe(str(wav_path), **opts)
    for seg in it:
        segments.append(STTSegment(start=float(seg.start), end=float(seg.end), text=seg.text.strip()))
    log(f"[STT] Done. Segments: {len(segments)} in {time.time()-t0:.1f}s")
    return segments

@dataclass
class TileEvent:
    t: float
    tile_idx: int

@dataclass
class TileIdentity:
    idx: int
    label_raw: str
    name_mapped: str

def hsv_blue_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([85, 70, 70])
    upper1 = np.array([130, 255, 255])
    lower2 = np.array([100, 80, 80])
    upper2 = np.array([140, 255, 255])
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    return cv2.bitwise_or(m1, m2)

def choose_grids() -> List[Tuple[int,int]]:
    return [(3,3), (4,4)]

def tile_border_ring_mask(h: int, w: int, rows: int, cols: int, border_px: int = 10) -> List[np.ndarray]:
    masks = []
    th = h // rows
    tw = w // cols
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r*th, min((r+1)*th, h)
            x0, x1 = c*tw, min((c+1)*tw, w)
            tile = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(tile, (x0, y0), (x1-1, y1-1), color=255, thickness=border_px)
            cv2.rectangle(tile, (x0+border_px, y0+border_px), (x1-1-border_px, y1-1-border_px), color=0, thickness=-1)
            masks.append(tile)
    return masks

def tile_bottom_label_roi(h: int, w: int, rows: int, cols: int, band_fraction: float=0.18) -> List[Tuple[int,int,int,int]]:
    rois = []
    th = h // rows
    tw = w // cols
    for r in range(rows):
        for c in range(cols):
            y0 = r*th + int(th*(1.0 - band_fraction))
            y1 = min((r+1)*th, h)
            x0 = c*tw
            x1 = min((c+1)*tw, w)
            rois.append((x0,y0,x1,y1))
    return rois

def detect_highlight_series(
    video_path: Path,
    fps_sample: float,
    do_ocr: bool,
    debug_snapshots: bool,
    snapshot_every: int,
    temp_dir: Path
) -> Tuple[List[TileEvent], Dict[int, TileIdentity], Tuple[int,int]]:
    log(f"[Visual] Starting analysis for: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    ok, probe = cap.read()
    if not ok or probe is None:
        cap.release()
        raise RuntimeError(f"Video opened but first frame read failed: {video_path}")
    v_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log(f"[Visual] Resolution: {v_w}x{v_h}, FPS: {v_fps:.2f}, Frames: {total_frames}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    step = max(1, int(round(v_fps / max(0.1, fps_sample))))
    grids = choose_grids()
    best_grid = None
    best_score = -1.0
    best_events = None
    best_labels = None

    for (R,C) in grids:
        log(f"[Visual] Trying grid {R}x{C}")
        masks = tile_border_ring_mask(v_h, v_w, R, C, border_px=8)
        rois = tile_bottom_label_roi(v_h, v_w, R, C, band_fraction=0.20)
        events: List[TileEvent] = []
        label_samples = {i: [] for i in range(R*C)}

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        processed = 0
        t0 = time.time()
        f_idx = 0
        snap_count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if f_idx % step != 0:
                f_idx += 1
                continue
            mask_blue = hsv_blue_mask(frame)
            scores = [int(cv2.countNonZero(cv2.bitwise_and(mask_blue, m))) for m in masks]
            tile_idx = int(np.argmax(scores))
            t = f_idx / v_fps
            events.append(TileEvent(t=t, tile_idx=tile_idx))

            if do_ocr and pytesseract is not None:
                (x0,y0,x1,y1) = rois[tile_idx]
                band = frame[y0:y1, x0:x1]
                band = cv2.resize(band, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
                gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                txt = pytesseract.image_to_string(gray, config="--psm 6").strip()
                txt = re.sub(r"\s+", " ", txt)
                if txt:
                    label_samples[tile_idx].append(txt)

            if debug_snapshots and (processed % max(1, snapshot_every) == 0):
                snap_path = temp_dir / f"debug_frame_{R}x{C}_{snap_count:04d}.jpg"
                try:
                    cv2.imwrite(str(snap_path), frame)
                    snap_count += 1
                except Exception as _:
                    pass

            f_idx += 1
            processed += 1
            if processed % 50 == 0:
                log(f"[Visual] Grid {R}x{C}: sampled {processed} frames...")

        same_runs = sum(1 for i in range(1, len(events)) if events[i].tile_idx == events[i-1].tile_idx)
        stability = same_runs / max(1, len(events))
        log(f"[Visual] Grid {R}x{C} stability={stability:.3f} (samples={len(events)}) in {time.time()-t0:.1f}s")
        if stability > best_score:
            best_score = stability
            best_grid = (R, C)
            best_events = events
            best_labels = label_samples

    identities: Dict[int, TileIdentity] = {}
    R, C = best_grid
    for idx in range(R*C):
        samples = best_labels.get(idx, [])
        if not samples:
            raw = ""
        else:
            values, counts = np.unique(np.array(samples), return_counts=True)
            raw = str(values[np.argmax(counts)])
        identities[idx] = TileIdentity(idx=idx, label_raw=raw, name_mapped="")

    cap.release()
    log(f"[Visual] Selected grid: {R}x{C} (stability={best_score:.3f})")
    return best_events, identities, best_grid

def first_stable_highlight_time(events: List[TileEvent], stable_len: int = 3) -> Optional[float]:
    if not events:
        return None
    for i in range(len(events) - stable_len + 1):
        tid = events[i].tile_idx
        if all(events[i+j].tile_idx == tid for j in range(1, stable_len)):
            return events[i].t
    return events[0].t

def refine_offset_grid(events: List[TileEvent], stt: List[STTSegment], initial_offset: float, search_window: float = 10.0, step: float = 0.1) -> float:
    change_times = [events[i].t for i in range(1, len(events)) if events[i].tile_idx != events[i-1].tile_idx]
    seg_starts = [s.start for s in stt]
    if not change_times or not seg_starts:
        return initial_offset

    def bin_times(times: List[float]) -> np.ndarray:
        if not times:
            return np.zeros(1, dtype=int)
        tmax = max(times) + 1.0
        n = int(math.ceil(tmax / 0.25))
        arr = np.zeros(n, dtype=int)
        for t in times:
            idx = min(n-1, int(round(t/0.25)))
            arr[idx] = 1
        return arr

    vis_bins = bin_times(change_times)
    best_off = initial_offset
    best_score = -1.0
    for off in np.arange(initial_offset - search_window, initial_offset + search_window + 1e-9, step):
        shifted = [t + off for t in seg_starts]
        stt_bins = bin_times(shifted)
        M = max(len(vis_bins), len(stt_bins))
        vb = np.pad(vis_bins, (0, M - len(vis_bins)))
        sb = np.pad(stt_bins, (0, M - len(stt_bins)))
        inter = np.sum((vb==1) & (sb==1))
        union = np.sum((vb==1) | (sb==1)) + 1e-6
        score = inter / union
        if score > best_score:
            best_score = score
            best_off = float(off)
    log(f"[Align] initial={initial_offset:.2f}s refined={best_off:.2f}s score={best_score:.3f}")
    return best_off

def map_identities_to_ics(identities: Dict[int, TileIdentity], attendees: List[Attendee]) -> Dict[int, TileIdentity]:
    icsonyms = [a.name for a in attendees if a.name]
    for idx, ident in identities.items():
        mapped = best_icsonym_match(ident.label_raw, icsonyms) if icsonyms else ident.label_raw
        identities[idx] = TileIdentity(idx=idx, label_raw=ident.label_raw, name_mapped=mapped or ident.label_raw)
    return identities

def merge_consecutive_segments(attributed: List[Tuple[str, float, float, str]]) -> List[Tuple[str, float, float, str]]:
    if not attributed:
        return []
    merged = [attributed[0]]
    for name, s, e, text in attributed[1:]:
        prev_name, ps, pe, ptext = merged[-1]
        if name == prev_name and abs(s - pe) < 0.75:
            merged[-1] = (prev_name, ps, e, (ptext + " " + text).strip())
        else:
            merged.append((name, s, e, text))
    return merged

def format_transcript_lines(attributed: List[Tuple[str, float, float, str]]) -> List[str]:
    return [f"{(name or 'Unknown')}: {text}" for name, s, e, text in attributed]

def ensure_work_dirs(base: Path) -> Tuple[Path, Path]:
    out_dir = base / "out"
    temp_dir = base / "temp"
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, temp_dir

def pick_inputs_with_gui(cfg: dict) -> dict:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    if cfg["VISUAL_VIDEO"] is None:
        cfg["VISUAL_VIDEO"] = Path(filedialog.askopenfilename(title="Select Zoom screen recording (visual only)",
                                                              filetypes=[("Video", "*.mp4 *.mov *.mkv *.avi *.webm")]))
    if cfg["AUDIO_SOURCE"] is None:
        cfg["AUDIO_SOURCE"] = Path(filedialog.askopenfilename(title="Select audio source (video w/ audio OR audio)",
                                                              filetypes=[("Media", "*.mp4 *.mov *.mkv *.avi *.webm *.wav *.m4a *.mp3 *.aac *.flac *.ogg *.opus")]))
    if cfg["ICS_FILE"] is None:
        cfg["ICS_FILE"] = Path(filedialog.askopenfilename(title="Select .ics invite",
                                                          filetypes=[("iCalendar", "*.ics")]))
    if cfg["WORK_DIR"] is None:
        cfg["WORK_DIR"] = Path(filedialog.askdirectory(title="Select Working Folder (will create out/ and temp/)"))
    root.destroy()
    return cfg

def write_outputs(
    vis_path: Path,
    aud_src: Path,
    ics_path: Optional[Path],
    out_dir: Path,
    temp_dir: Path,
    lines: List[str],
    attendees: List[Attendee],
    identities: Dict[int, TileIdentity],
    best_offset: float,
    stt_segments_count: int,
    grid_info: Optional[Tuple[int,int]],
    status: str
) -> Tuple[Path, Path]:
    base = vis_path.stem + "_fused"
    out_txt = out_dir / f"{base}_attributed_transcript.txt"
    out_json = out_dir / f"{base}_diagnostics.json"
    out_txt.write_text("\n".join(lines), encoding="utf-8")

    diag = {
        "status": status,  # "success" or "stt_only"
        "visual_video": str(vis_path),
        "audio_source": str(aud_src),
        "ics": str(ics_path) if ics_path else "",
        "events_samples": None if grid_info is None else "see grid",
        "attendees": [a.__dict__ for a in attendees],
        "identities": {k: identities[k].__dict__ for k in identities} if identities else {},
        "offset_seconds": float(best_offset),
        "stt_segments": int(stt_segments_count),
        "grid": ({"rows": int(grid_info[0]), "cols": int(grid_info[1])} if grid_info else {}),
        "work_dirs": {"out": str(out_dir), "temp": str(temp_dir)},
    }
    out_json.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    log(f"[OUT] Transcript:  {out_txt}")
    log(f"[OUT] Diagnostics: {out_json}")
    return out_txt, out_json

def main(argv=None):
    # Optional flag: --nogui to skip GUI even if USE_GUI_DEFAULT is True
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--nogui", action="store_true")
    args, _ = parser.parse_known_args(argv)

    cfg = dict(CONFIG)
    use_gui = USE_GUI_DEFAULT and not args.nogui

    if use_gui:
        cfg = pick_inputs_with_gui(cfg)

    vis_path = Path(cfg["VISUAL_VIDEO"]) if cfg["VISUAL_VIDEO"] else None
    aud_src  = Path(cfg["AUDIO_SOURCE"]) if cfg["AUDIO_SOURCE"] else None
    ics_path = Path(cfg["ICS_FILE"]) if cfg["ICS_FILE"] else None
    work_dir = Path(cfg["WORK_DIR"]) if cfg["WORK_DIR"] else None

    if not vis_path or not vis_path.exists():
        raise FileNotFoundError("Missing VISUAL_VIDEO. Set CONFIG['VISUAL_VIDEO'] or use GUI.")
    if not aud_src or not aud_src.exists():
        raise FileNotFoundError("Missing AUDIO_SOURCE. Set CONFIG['AUDIO_SOURCE'] or use GUI.")
    if not work_dir:
        raise FileNotFoundError("Missing WORK_DIR. Set CONFIG['WORK_DIR'] or use GUI.")

    out_dir, temp_dir = ensure_work_dirs(work_dir)
    log(f"[Paths] Work: {work_dir}")
    log(f"[Paths] Out:  {out_dir}")
    log(f"[Paths] Temp: {temp_dir}")

    attendees = []
    if ics_path and ics_path.exists():
        attendees = parse_ics_attendees(ics_path)
        log(f"[ICS] Loaded attendees: {len(attendees)}")
    else:
        log("[ICS] No .ics provided; will use OCR labels directly.")

    # Audio → WAV
    wav_path = temp_dir / (aud_src.stem + "_audio16k.wav")
    log(f"[Audio] Extracting/normalizing to WAV: {wav_path}")
    t0 = time.time()
    wav_path = extract_audio_to_wav(aud_src, wav_path, sr=16000)
    log(f"[Audio] WAV ready in {time.time()-t0:.1f}s")

    # STT
    stt_segments = transcribe_wav_fwhisper(
        wav_path,
        model_size=cfg["WHISPER_MODEL"],
        device=cfg["WHISPER_DEVICE"],
        compute_type=cfg["WHISPER_COMPUTE"]
    )
    if not stt_segments:
        log("[STT] No segments produced—writing empty transcript and exiting.")
        write_outputs(vis_path, aud_src, ics_path, out_dir, temp_dir, [], attendees, {}, 0.0, 0, None, status="stt_only")
        close_log()
        return 2

    # VISUAL Preflight
    log("[Visual] Preflight: opening video...")
    test_cap = cv2.VideoCapture(str(vis_path))
    if not test_cap.isOpened():
        log(f"[Visual][ERROR] Cannot open video: {vis_path}")
        lines = [f"Unknown: {s.text}" for s in stt_segments]
        write_outputs(vis_path, aud_src, ics_path, out_dir, temp_dir, lines, attendees, {}, 0.0, len(stt_segments), None, status="stt_only")
        close_log()
        return 4
    ok, frm0 = test_cap.read()
    test_cap.release()
    if not ok or frm0 is None:
        log(f"[Visual][ERROR] Opened but first frame read failed: {vis_path}")
        lines = [f"Unknown: {s.text}" for s in stt_segments]
        write_outputs(vis_path, aud_src, ics_path, out_dir, temp_dir, lines, attendees, {}, 0.0, len(stt_segments), None, status="stt_only")
        close_log()
        return 5
    log("[Visual] Preflight OK; starting detection...")

    # VISUAL Detection
    identities: Dict[int, TileIdentity] = {}
    events: List[TileEvent] = []
    grid_info: Optional[Tuple[int,int]] = None
    try:
        events, identities, grid_info = detect_highlight_series(
            vis_path,
            fps_sample=float(cfg["FPS_SAMPLE"]),
            do_ocr=bool(cfg["OCR_ENABLED"] and pytesseract is not None),
            debug_snapshots=bool(cfg["DEBUG_SNAPSHOTS"]),
            snapshot_every=int(cfg["SNAPSHOT_EVERY"]),
            temp_dir=temp_dir
        )
    except Exception as ex:
        elog("[Visual][FATAL] Exception in detection: " + str(ex))
        traceback.print_exc()
        # Fallback to STT-only output
        lines = [f"Unknown: {s.text}" for s in stt_segments]
        write_outputs(vis_path, aud_src, ics_path, out_dir, temp_dir, lines, attendees, {}, 0.0, len(stt_segments), None, status="stt_only")
        close_log()
        return 6

    if not events:
        log("[Visual] No highlight events detected—writing STT-only transcript.")
        lines = [f"Unknown: {s.text}" for s in stt_segments]
        write_outputs(vis_path, aud_src, ics_path, out_dir, temp_dir, lines, attendees, {}, 0.0, len(stt_segments), grid_info, status="stt_only")
        close_log()
        return 3

    R, C = grid_info
    log(f"[Visual] Events: {len(events)}, Grid: {R}x{C}")

    # Map OCR -> ICS
    identities = map_identities_to_ics(identities, attendees)

    # Offset estimate + refine
    t_vis0 = first_stable_highlight_time(events) or 0.0
    t_aud0 = stt_segments[0].start if stt_segments else 0.0
    initial = t_vis0 - t_aud0
    log(f"[Align] Initial guess offset = visual({t_vis0:.2f}) - audio({t_aud0:.2f}) = {initial:.2f}s")
    best_offset = refine_offset_grid(events, stt_segments, initial_offset=initial, search_window=10.0, step=0.1)

    # Attribute by nearest visual sample at seg.start + offset
    times_arr = np.array([ev.t for ev in events], dtype=np.float32)
    tiles_arr = np.array([ev.tile_idx for ev in events], dtype=np.int16)

    attributed: List[Tuple[str, float, float, str]] = []
    for seg in stt_segments:
        t_vis = seg.start + best_offset
        idx = np.searchsorted(times_arr, t_vis, side="right") - 1
        idx = int(np.clip(idx, 0, len(times_arr)-1))
        tidx = int(tiles_arr[idx])
        ident = identities.get(tidx)
        name = (ident.name_mapped or ident.label_raw or f"Tile{tidx+1}") if ident else f"Tile{tidx+1}"
        attributed.append((name, seg.start, seg.end, seg.text))

    attributed = merge_consecutive_segments(attributed)
    lines = format_transcript_lines(attributed)

    # Write outputs
    base = vis_path.stem + "_fused"
    out_txt = out_dir / f"{base}_attributed_transcript.txt"
    out_json = out_dir / f"{base}_diagnostics.json"
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    diag = {
        "visual_video": str(vis_path),
        "audio_source": str(aud_src),
        "ics": str(ics_path) if ics_path else "",
        "events_samples": len(events),
        "attendees": [a.__dict__ for a in attendees],
        "identities": {k: identities[k].__dict__ for k in identities},
        "offset_seconds": float(best_offset),
        "stt_segments": len(stt_segments),
        "grid": {"rows": int(R), "cols": int(C)},
        "work_dirs": {"out": str(out_txt.parent), "temp": str((out_txt.parent.parent / "temp"))},
    }
    out_json.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    log(f"[OUT] Transcript:  {out_txt}")
    log(f"[OUT] Diagnostics: {out_json}")
    log(f"[DONE] Attributed lines: {len(lines)}")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as ex:
        elog("[FATAL] Unhandled exception: " + str(ex))
        traceback.print_exc()
        sys.exit(1)
