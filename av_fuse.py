#!/usr/bin/env python3
"""
av_fuse_attributor.py

Purpose
-------
Fuse *visual attribution* from a clean Zoom screen-recording of a Teams meeting
(with visible active-speaker blue outline) with *audio* from a separate device
(e.g., phone video or standalone audio file), then generate a single attributed
transcript (Speaker: text).

This tool is designed to live alongside Offline_Meeting_Attributor/attributor.py
and reuses the same ideas (HSV-based blue-border detection, grid heuristics,
.ics attendee parsing + fuzzy-name match, STT via faster-whisper). It does NOT
modify any of your existing files.

Inputs
------
- Visual video (no audio needed): Zoom screen recording of the Teams call.
- Audio source: either (a) a different video that *has* audio, or (b) an audio file.
- .ICS invite: to normalize names/emails/companies.

If CLI flags are omitted, a small GUI file picker will prompt for files.

Outputs
-------
- <basename>_attributed_transcript.txt  (lines like "Name: ...")
- <basename>_diagnostics.json            (alignment + metadata, optional)
- Optional intermediate WAV extracted from the audio source (temp folder).

Dependencies
------------
pip install faster-whisper opencv-python numpy pytesseract rapidfuzz
Also requires:
- FFmpeg available on PATH
- Tesseract-OCR installed and on PATH (for pytesseract), or set TESSDATA_PREFIX.

Notes
-----
- We detect the active speaker by scanning for Teams' blue/cyan outline within a
  per-tile "border ring". We auto-try 3x3 and 4x4 grids and pick the layout with
  the best border-signal stability.
- We OCR tile bottom labels to map tiles → names; then fuzzy-map to .ics names.
- AV sync: we estimate an initial offset using the first stable highlight and the
  first STT segment, then refine by grid-search over ±10s to align highlight-change
  times with segment starts.
- Final text merges consecutive segments from the same attributed speaker.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

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
    # Leave as None to trigger GUI file pickers
    "VISUAL_VIDEO": None,  # Zoom screen-record (visual only)
    "AUDIO_SOURCE": None,  # Phone video/audio (with audio) OR audio file
    "ICS_FILE": None,      # .ics invite
    "WORK_DIR": None,      # Folder to hold out\ and temp\ (GUI will prompt if None)

    # STT / model
    "WHISPER_MODEL": "medium",
    "WHISPER_DEVICE": "cuda",      # "cuda" or "cpu"
    "WHISPER_COMPUTE": "float16",  # "float16", "int8", ...

    # Visual detection
    "FPS_SAMPLE": 2.0,      # frames/sec to sample for border detection
    "OCR_ENABLED": True,    # False to disable OCR of bottom labels
}

USE_GUI_DEFAULT = True  # set to False to rely solely on CONFIG (and run with --nogui if desired)

# ==========================
# Logging utilities
# ==========================
def log(msg: str):
    print(msg, flush=True)

def elog(msg: str):
    print(msg, file=sys.stderr, flush=True)

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
            name = name.replace(r"\\,", ",").strip('"')
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
    s = re.sub(r"[^A-Za-z0-9@.\\-' ]+", "", raw).strip()
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

def extract_audio_to_wav(src: Path, out_dir: Path, sr: int = 16000) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav = out_dir / (src.stem + "_audio16k.wav")
    cmd = ["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", str(out_wav)]
    proc = run(cmd, check=True)
    if proc.returncode != 0:
        eprint(proc.stderr)
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

def choose_grid(frame_shape: Tuple[int,int,int]) -> List[Tuple[int,int]]:
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

def detect_highlight_series(video_path: Path, fps_sample: float = 2.0, grids=((3,3),(4,4)), ocr: bool=True) -> Tuple[List[TileEvent], Dict[int, TileIdentity]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    v_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(round(v_fps / fps_sample)))
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Empty video.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    best_grid = None
    best_score = -1.0
    blue_events_by_grid = {}
    label_texts_by_grid = {}
    for (R,C) in grids:
        masks = tile_border_ring_mask(v_h, v_w, R, C, border_px=8)
        rois = tile_bottom_label_roi(v_h, v_w, R, C, band_fraction=0.20)
        events = []
        label_samples = {i: [] for i in range(R*C)}
        f_idx = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if f_idx % step != 0:
                f_idx += 1
                continue
            mask_blue = hsv_blue_mask(frame)
            scores = []
            for i, m in enumerate(masks):
                s = int(cv2.countNonZero(cv2.bitwise_and(mask_blue, m)))
                scores.append(s)
            tile_idx = int(np.argmax(scores))
            t = f_idx / v_fps
            events.append(TileEvent(t=t, tile_idx=tile_idx))
            if ocr and pytesseract is not None:
                (x0,y0,x1,y1) = rois[tile_idx]
                band = frame[y0:y1, x0:x1]
                band = cv2.resize(band, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
                gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                txt = pytesseract.image_to_string(gray, config="--psm 6").strip()
                txt = re.sub(r"\s+", " ", txt)
                if txt:
                    label_samples[tile_idx].append(txt)
            f_idx += 1
        same_runs = 0
        for i in range(1, len(events)):
            if events[i].tile_idx == events[i-1].tile_idx:
                same_runs += 1
        stability = same_runs / max(1, len(events))
        if stability > best_score:
            best_score = stability
            best_grid = (R, C)
            blue_events_by_grid[(R,C)] = events
            label_texts_by_grid[(R,C)] = label_samples
    events = blue_events_by_grid[best_grid]
    label_samples = label_texts_by_grid[best_grid]
    R, C = best_grid
    eprint(f"[Visual] Selected grid: {R}x{C} (stability={best_score:.3f})")
    identities: Dict[int, TileIdentity] = {}
    for idx in range(R*C):
        samples = label_samples.get(idx, [])
        if not samples:
            raw = ""
        else:
            values, counts = np.unique(np.array(samples), return_counts=True)
            raw = str(values[np.argmax(counts)])
        identities[idx] = TileIdentity(idx=idx, label_raw=raw, name_mapped="")
    cap.release()
    return events, identities

def first_stable_highlight_time(events: List[TileEvent], stable_len: int = 3) -> Optional[float]:
    if not events:
        return None
    for i in range(len(events) - stable_len + 1):
        tid = events[i].tile_idx
        ok = True
        for j in range(1, stable_len):
            if events[i+j].tile_idx != tid:
                ok = False
                break
        if ok:
            return events[i].t
    return events[0].t

def refine_offset_grid(events: List[TileEvent], stt: List[STTSegment], initial_offset: float, search_window: float = 10.0, step: float = 0.1) -> float:
    change_times = []
    for i in range(1, len(events)):
        if events[i].tile_idx != events[i-1].tile_idx:
            change_times.append(events[i].t)
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
    offs = np.arange(initial_offset - search_window, initial_offset + search_window + 1e-9, step)
    for off in offs:
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
    eprint(f"[Align] initial={initial_offset:.2f}s refined={best_off:.2f}s score={best_score:.3f}")
    return best_off

def map_identities_to_ics(identities: Dict[int, TileIdentity], attendees: List[Attendee]) -> Dict[int, TileIdentity]:
    icsonyms = [a.name for a in attendees if a.name]
    for idx, ident in identities.items():
        mapped = best_icsonym_match(ident.label_raw, icsonyms) if icsonyms else ident.label_raw
        identities[idx] = TileIdentity(idx=idx, label_raw=ident.label_raw, name_mapped=mapped or ident.label_raw)
    return identities

def build_tile_lookup(events: List[TileEvent]) -> Tuple[np.ndarray, np.ndarray]:
    times = np.array([ev.t for ev in events], dtype=np.float32)
    tiles = np.array([ev.tile_idx for ev in events], dtype=np.int16)
    return times, tiles

def tile_at_time(times: np.ndarray, tiles: np.ndarray, t: float) -> Optional[int]:
    if times.size == 0:
        return None
    idx = np.searchsorted(times, t, side="right") - 1
    idx = np.clip(idx, 0, len(times)-1)
    return int(tiles[idx])

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
    lines = []
    for name, s, e, text in attributed:
        nm = name if name else "Unknown"
        lines.append(f"{nm}: {text}")
    return lines

def main(argv=None):
    parser = argparse.ArgumentParser(description="Fuse visual attribution with separate audio to produce an attributed transcript.")
    parser.add_argument("--visual-video", type=Path, help="Path to the clean Zoom screen recording (Teams view).")
    parser.add_argument("--audio-source", type=Path, help="Path to audio source (video with audio OR audio file).")
    parser.add_argument("--ics", type=Path, help="Path to .ics invite (attendees).")
    parser.add_argument("--outdir", type=Path, default=Path("."), help="Directory for outputs.")
    parser.add_argument("--model", default="medium", help="faster-whisper model size (e.g., small, medium, large-v3).")
    parser.add_argument("--device", default="cuda", help="faster-whisper device (cuda/cpu).")
    parser.add_argument("--compute-type", default="float16", help="faster-whisper compute type (float16/int8).")
    parser.add_argument("--fps-sample", type=float, default=2.0, help="FPS to sample visual frames for border detection.")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR of name labels.")
    parser.add_argument("--offset", type=float, default=None, help="Override AV offset in seconds (visual_t - audio_t).")
    parser.add_argument("--dry-run", action="store_true", help="Run detection/alignment but do not write transcript.")
    parser.add_argument("--diagnostics", action="store_true", help="Write diagnostics JSON.")
    parser.add_argument("--gui", action="store_true", help="Force file-pickers even if flags provided.")
    args = parser.parse_args(argv)

    vis_path = args.visual_video
    aud_src = args.audio_source
    ics_path = args.ics

    if args.gui or vis_path is None or aud_src is None or ics_path is None:
        try:
            import tkinter as tk
            from tkinter import filedialog
            tk.Tk().withdraw()
            if vis_path is None:
                vis_path = Path(filedialog.askopenfilename(title="Select Zoom screen recording (visual only)", filetypes=[("Video", "*.mp4 *.mov *.mkv *.avi *.webm")]))
            if aud_src is None:
                aud_src = Path(filedialog.askopenfilename(title="Select audio source (video w/ audio OR audio)", filetypes=[("Media", "*.mp4 *.mov *.mkv *.avi *.webm *.wav *.m4a *.mp3 *.aac *.flac *.ogg *.opus")]))
            if ics_path is None:
                ics_path = Path(filedialog.askopenfilename(title="Select .ics invite", filetypes=[("iCalendar", "*.ics")]))
        except Exception as ex:
            eprint("[GUI] File dialog failed; ensure --visual-video, --audio-source, --ics are provided.")
            raise

    if not vis_path or not vis_path.exists():
        raise FileNotFoundError("--visual-video not provided or missing.")
    if not aud_src or not aud_src.exists():
        raise FileNotFoundError("--audio-source not provided or missing.")
    if not ics_path or not ics_path.exists():
        eprint("[WARN] --ics missing or not found; proceeding without attendee normalization.")
        attendees = []
    else:
        attendees = parse_ics_attendees(ics_path)
        eprint(f"[ICS] Loaded attendees: {len(attendees)}")

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        if is_audio_file(aud_src):
            wav_path = extract_audio_to_wav(aud_src, td)
        elif is_video_file(aud_src):
            wav_path = extract_audio_to_wav(aud_src, td)
        else:
            raise ValueError("audio-source must be a video with audio or an audio file.")
        stt_segments = transcribe_wav_fwhisper(wav_path, model_size=args.model, device=args.device, compute_type=args.compute_type)

    grids = choose_grid((0,0,0))
    events, identities = detect_highlight_series(vis_path, fps_sample=args.fps_sample, grids=grids, ocr=(not args.no_ocr))
    identities = map_identities_to_ics(identities, attendees)
    times_arr, tiles_arr = build_tile_lookup(events)

    if args.offset is not None:
        best_offset = args.offset
        eprint(f"[Align] Using user-provided offset: {best_offset:.2f}s")
    else:
        t_vis0 = first_stable_highlight_time(events) or 0.0
        t_aud0 = stt_segments[0].start if stt_segments else 0.0
        initial = t_vis0 - t_aud0
        best_offset = refine_offset_grid(events, stt_segments, initial_offset=initial, search_window=10.0, step=0.1)

    attributed: List[Tuple[str, float, float, str]] = []
    for seg in stt_segments:
        t_vis = seg.start + best_offset
        tidx = tile_at_time(times_arr, tiles_arr, t_vis)
        if tidx is None:
            name = "Unknown"
        else:
            ident = identities.get(tidx)
            name = (ident.name_mapped or ident.label_raw or f"Tile{tidx+1}") if ident else f"Tile{tidx+1}"
        attributed.append((name, seg.start, seg.end, seg.text))

    attributed = merge_consecutive_segments(attributed)
    lines = format_transcript_lines(attributed)

    base = (vis_path.stem + "_fused")
    out_txt = outdir / f"{base}_attributed_transcript.txt"
    if not args.dry_run:
        out_txt.write_text("\n".join(lines), encoding="utf-8")
        eprint(f"[OUT] Wrote transcript: {out_txt} ({len(lines)} lines)")

    if args.diagnostics:
        diag = {
            "visual_video": str(vis_path),
            "audio_source": str(aud_src),
            "ics": str(ics_path) if ics_path else "",
            "events_samples": len(events),
            "attendees": [a.__dict__ for a in attendees],
            "identities": {k: identities[k].__dict__ for k in identities},
            "offset_seconds": best_offset,
            "stt_segments": len(stt_segments),
            "grid_options": [(3,3),(4,4)],
        }
        out_json = outdir / f"{base}_diagnostics.json"
        out_json.write_text(json.dumps(diag, indent=2), encoding="utf-8")
        eprint(f"[OUT] Wrote diagnostics: {out_json}")

    if args.dry_run:
        eprint("[DRY-RUN] Completed without writing transcript.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        eprint("Interrupted by user.")
        sys.exit(130)
    except Exception as ex:
        eprint(f"[ERROR] {ex}")
        sys.exit(1)
