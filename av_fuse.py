#!/usr/bin/env python3
# av_fuse.py — V1.0.8
#
# Changes vs V1.0.7:
# - Normalize ICS attendee names from emails (e.g., "first.last@..." → "First Last").
# - Add CONFIG["TILE_NAME_OVERRIDES"] to force-map a tile index to a display name.
# - Emit a grid preview image with tile indices, and a tile-activity CSV.
# - Keep OCR name sanitation + strict ICS mapping from V1.0.7.
# - Preserve provisional writes + atexit emergency writer safeguards.
#
from __future__ import annotations

import argparse, atexit, json, math, os, re, subprocess, sys, time, traceback, gc, csv
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

CONFIG = {
    "VISUAL_VIDEO": None,
    "AUDIO_SOURCE": None,
    "ICS_FILE": None,
    "WORK_DIR": None,

    "WHISPER_MODEL": "medium",
    "WHISPER_DEVICE": "cuda",      # set to "cpu" if you want to test without GPU
    "WHISPER_COMPUTE": "float16",  # "float16" or "int8" on CPU

    "FPS_SAMPLE": 2.0,
    "OCR_ENABLED": True,

    "DEBUG_SNAPSHOTS": False,
    "SNAPSHOT_EVERY": 150,

    # NEW: manual tile -> name overrides (use tile index, zero-based)
    "TILE_NAME_OVERRIDES": {
        # 14: "Tyler Lastname",
        # 7: "Justin Lastname",
    },

    # NEW: crop away UI bars (pixels). Tune if needed.
    "CANVAS_CROP": {"top": 80, "bottom": 80, "left": 0, "right": 0},

    # NEW: optionally force the grid layout. Set to (3,3) for Teams gallery.
    # Leave as None to auto-try both.
    "FORCE_GRID": None,  # e.g., (3,3)
}
# Optional: force Teams layout for this run
CONFIG["FORCE_GRID"] = (3, 3)

USE_GUI_DEFAULT = True

# ----------------- logging -----------------
_log_file_handle = None
def _open_log_file(path: Path):
    global _log_file_handle
    try:
        _log_file_handle = path.open("a", encoding="utf-8")
        _log_file_handle.write(f"===== RUN START {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n"); _log_file_handle.flush()
    except Exception:
        _log_file_handle = None

def _write_log_file(msg: str):
    if _log_file_handle:
        try:
            _log_file_handle.write(msg + "\n"); _log_file_handle.flush()
        except Exception:
            pass

def log(msg: str):
    print(msg, flush=True); _write_log_file(msg)

def elog(msg: str):
    print(msg, file=sys.stderr, flush=True); _write_log_file("[ERR] " + msg)

def close_log():
    global _log_file_handle
    if _log_file_handle:
        try:
            _log_file_handle.write(f"===== RUN END {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            _log_file_handle.flush(); _log_file_handle.close()
        except Exception:
            pass
        _log_file_handle = None

def run(cmd: List[str], check=True) -> subprocess.CompletedProcess:
    elog("> " + " ".join(cmd))
    return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# ----------------- helpers -----------------
def is_audio_file(path: Path) -> bool:
    return path.suffix.lower() in {".wav", ".m4a", ".mp3", ".aac", ".flac", ".ogg", ".opus"}

def is_mono_16k_wav(path: Path) -> bool:
    if path.suffix.lower() != ".wav": return False
    if "audio16k" in path.stem.lower(): return True
    try:
        proc = subprocess.run(
            ["ffprobe","-v","error","-select_streams","a:0","-show_entries","stream=sample_rate,channels",
             "-of","default=nw=1:nk=1", str(path)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
        if len(lines) >= 2:
            return (int(lines[0]) == 16000 and int(lines[1]) == 1)
    except Exception:
        pass
    return False

@dataclass
class Attendee:
    name: str
    email: str = ""
    company: str = ""

def _crop_canvas(frame: np.ndarray, crop: Dict[str,int]) -> Tuple[np.ndarray, Tuple[int,int]]:
    t = int(crop.get("top", 0)); b = int(crop.get("bottom", 0))
    l = int(crop.get("left", 0)); r = int(crop.get("right", 0))
    h, w = frame.shape[:2]
    y0 = min(max(0, t), h-1); y1 = max(y0+1, h - max(0, b))
    x0 = min(max(0, l), w-1); x1 = max(x0+1, w - max(0, r))
    return frame[y0:y1, x0:x1].copy(), (x0, y0)

def highlight_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Border mask that works for both Zoom (blue) and Teams (white).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # blue-ish ring (Zoom)
    lower1 = np.array([85, 70, 70]); upper1 = np.array([130, 255, 255])
    lower2 = np.array([100, 80, 80]); upper2 = np.array([140, 255, 255])
    blue = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
    # white/bright ring (Teams)
    v = hsv[:, :, 2]; s = hsv[:, :, 1]
    bright = cv2.inRange(v, 220, 255)
    low_sat = cv2.inRange(s, 0, 60)
    white = cv2.bitwise_and(bright, low_sat)
    return cv2.bitwise_or(blue, white)

# NEW: derive a human name from an email local part (first.last → First Last)
def _name_from_email(addr: str) -> str:
    """
    Convert 'first.last@domain' into 'First Last'. Returns '' if cannot parse.
    """
    if not addr or "@" not in addr:
        return ""
    local = addr.split("@", 1)[0]
    parts = re.split(r"[._+\-]+", local)
    parts = [p for p in parts if p and p.isalpha()]
    if len(parts) < 2:
        return ""
    return " ".join(w.capitalize() for w in parts[:3])

def parse_ics_attendees(p: Path) -> List[Attendee]:
    out: List[Attendee] = []
    if not p or not p.exists(): return out
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    org = ""
    for ln in lines:
        L = ln.strip()
        if L.upper().startswith("ATTENDEE"):
            m1 = re.search(r";CN=([^;:]+)", L, flags=re.I)
            m2 = re.search(r":mailto:([^ \r\n]+)", L, flags=re.I)
            name = (m1.group(1).strip() if m1 else "").replace(r"\,", ",").strip('"')
            email = (m2.group(1).strip() if m2 else "")
            # Normalize CN when blank or looks like an email
            if (not name) or ("@" in name):
                derived = _name_from_email(email or name)
                if derived:
                    name = derived
            out.append(Attendee(name=name, email=email, company=""))
        elif L.upper().startswith("ORGANIZATION:") or L.upper().startswith("ORG:"):
            org = L.split(":",1)[-1].strip()
    if org:
        for a in out:
            if not a.company: a.company = org
    return out

def best_icsonym_match(raw: str, icsonyms: List[str]) -> Optional[str]:
    if not raw: return None
    s = re.sub(r"[^A-Za-z0-9@\.\-\' ]+", "", raw).strip()
    if not s: return None
    if not icsonyms: return s
    if process and fuzz:
        m = process.extractOne(s, icsonyms, scorer=fuzz.WRatio, score_cutoff=72)
        return m[0] if m else s
    for can in icsonyms:
        if s.lower() == can.lower(): return can
    return s

# --- Name sanity filters for OCR → ICS mapping ---
STOPWORDS = {
    "inbox", "zoom", "workplace", "screen", "sharing", "snipping", "tool",
    "client", "secure", "entry", "external", "meeting", "recording",
    "cod", "windows", "mail", "outlook"
}

def _is_plausible_person_name(txt: str) -> bool:
    if not txt:
        return False
    s = txt.strip()
    low = s.lower()
    if any(w in low for w in STOPWORDS):
        return False
    if "@" in s or "|" in s or "#" in s:
        return False
    # keep only letters, spaces, hyphens, apostrophes
    cleaned = re.sub(r"[^A-Za-z \-']", " ", s)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return False
    tokens = cleaned.split()
    # typical display names are 2–3 tokens
    if len(tokens) < 2 or len(tokens) > 3:
        return False
    # require tokens to be capitalized words (e.g., "John", "O'Neil")
    cap_like = sum(1 for t in tokens if re.match(r"^[A-Z][a-z'\-]+$", t))
    return cap_like >= 2

def clean_ocr_name(raw: str) -> str:
    """Return a cleaned candidate name or '' if it's not plausibly a person name."""
    if not raw:
        return ""
    s = re.sub(r"\s+", " ", raw).strip()
    if not _is_plausible_person_name(s):
        return ""
    s = re.sub(r"[^A-Za-z \-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join(w[0].upper() + w[1:] if w else w for w in s.split(" "))

def strict_map_to_ics(ocr_name: str, attendees: List[Attendee], cutoff: int = 88) -> str:
    """
    Map a plausible OCR name to an ICS attendee only if:
      - fuzzy score is high (>= cutoff), AND
      - both first and last tokens from OCR appear in the ICS name.
    Otherwise return '' (decline mapping).
    """
    if not ocr_name or not attendees:
        return ""
    icsonyms = [a.name for a in attendees if a.name]
    if not icsonyms:
        return ""
    toks = [t for t in ocr_name.split() if len(t) >= 2]
    if len(toks) < 2:
        return ""
    first, last = toks[0].lower(), toks[-1].lower()

    best = None
    best_score = -1
    if process and fuzz:
        cand, score, _ = process.extractOne(ocr_name, icsonyms, scorer=fuzz.WRatio)
        best, best_score = cand, score
    else:
        for cand in icsonyms:
            if ocr_name.lower() == cand.lower():
                best, best_score = cand, 100
                break

    if not best or best_score < cutoff:
        return ""

    btoks = [t.lower() for t in re.findall(r"[A-Za-z]+", best)]
    if first in btoks and last in btoks:
        return best
    return ""

def extract_audio_to_wav(src: Path, out_wav: Path, sr: int = 16000) -> Path:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    proc = run(["ffmpeg","-y","-i",str(src),"-vn","-ac","1","-ar",str(sr),"-f","wav",str(out_wav)], check=True)
    if proc.returncode != 0:
        elog(proc.stderr); raise RuntimeError("ffmpeg failed")
    return out_wav

@dataclass
class STTSegment:
    start: float
    end: float
    text: str

# --------- GLOBAL Whisper model cache ----------
_WHISPER_MODEL = None
def _get_whisper(model_size: str, device: str, compute_type: str):
    global _WHISPER_MODEL
    if WhisperModel is None:
        raise ImportError("faster-whisper not installed")
    if _WHISPER_MODEL is None:
        log(f"[STT] Loading Whisper model once: {model_size} ({device}/{compute_type})")
        _WHISPER_MODEL = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _WHISPER_MODEL

def transcribe_wav_fwhisper(wav_path: Path, model_size: str, device: str, compute_type: str) -> List[STTSegment]:
    model = _get_whisper(model_size, device, compute_type)
    segments: List[STTSegment] = []
    opts = dict(beam_size=5, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
    t0 = time.time()
    it, _info = model.transcribe(str(wav_path), **opts)
    for seg in it:
        segments.append(STTSegment(float(seg.start), float(seg.end), seg.text.strip()))
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
    lower1 = np.array([85, 70, 70]); upper1 = np.array([130, 255, 255])
    lower2 = np.array([100, 80, 80]); upper2 = np.array([140, 255, 255])
    return cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))

def choose_grids() -> List[Tuple[int,int]]:
    fg = CONFIG.get("FORCE_GRID")
    if isinstance(fg, (tuple, list)) and len(fg) == 2:
        return [tuple(fg)]
    return [(3,3), (4,4)]

def tile_border_ring_mask(h: int, w: int, rows: int, cols: int, border_px: int = 10) -> List[np.ndarray]:
    masks = []; th = h // rows; tw = w // cols
    for r in range(rows):
        for c in range(cols):
            y0,y1 = r*th, min((r+1)*th, h); x0,x1 = c*tw, min((c+1)*tw, w)
            tile = np.zeros((h,w), dtype=np.uint8)
            cv2.rectangle(tile,(x0,y0),(x1-1,y1-1),255,border_px)
            cv2.rectangle(tile,(x0+border_px,y0+border_px),(x1-1-border_px,y1-1-border_px),0,-1)
            masks.append(tile)
    return masks

def tile_bottom_label_roi(h: int, w: int, rows: int, cols: int, band_fraction: float=0.18):
    rois = []; th = h // rows; tw = w // cols
    for r in range(rows):
        for c in range(cols):
            y0 = r*th + int(th*(1.0 - band_fraction)); y1 = min((r+1)*th, h)
            x0 = c*tw; x1 = min((c+1)*tw, w); rois.append((x0,y0,x1,y1))
    return rois

def _open_video_with_backoffs(path: Path):
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened(): return cap
    cap2 = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if cap2.isOpened(): return cap2
    return None

def detect_highlight_series(video_path: Path, fps_sample: float, do_ocr: bool,
                            debug_snapshots: bool, snapshot_every: int, temp_dir: Path):
    log(f"[Visual] Starting analysis for: {video_path}")
    cap = _open_video_with_backoffs(video_path)
    if cap is None or not cap.isOpened():
        raise RuntimeError(f"Could not open video (CAP_ANY/FFMPEG failed): {video_path}")
    ok, probe = cap.read()
    if not ok or probe is None:
        cap.release(); raise RuntimeError("Opened video, but first frame read failed")
    v_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log(f"[Visual] Resolution: {v_w}x{v_h}, FPS: {v_fps:.2f}, Frames: {total_frames}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    step = max(1, int(round(v_fps / max(0.1, fps_sample))))
    grids = choose_grids()
    best_grid=None; best_score=-1.0; best_events=None; best_labels=None

    for (R,C) in grids:
        log(f"[Visual] Trying grid {R}x{C}")
        # read one fresh frame to size the cropped canvas
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok_probe, probe = cap.read()
        if not ok_probe or probe is None:
            cap.release(); raise RuntimeError("Opened video, but probe read failed")
        probe_cropped, _off = _crop_canvas(probe, CONFIG.get("CANVAS_CROP", {}))
        v_h, v_w = probe_cropped.shape[:2]

        masks = tile_border_ring_mask(v_h, v_w, R, C, border_px=8)
        rois = tile_bottom_label_roi(v_h, v_w, R, C, band_fraction=0.20)

        events: List[TileEvent] = []; label_samples = {i: [] for i in range(R*C)}
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        processed=0; f_idx=0; t0=time.time(); snap=0

        while True:
            ok, frame = cap.read()
            if not ok: break
            if f_idx % step != 0:
                f_idx += 1; continue

            # NEW: crop UI bars before analysis
            frame_c, _off = _crop_canvas(frame, CONFIG.get("CANVAS_CROP", {}))
            mask_bord = highlight_mask(frame_c)

            # score per tile ring
            scores = [int(cv2.countNonZero(cv2.bitwise_and(mask_bord, m))) for m in masks]
            tile_idx = int(np.argmax(scores)); t = f_idx / v_fps
            events.append(TileEvent(t, tile_idx))

            if do_ocr and pytesseract is not None:
                x0,y0,x1,y1 = rois[tile_idx]
                band = frame_c[y0:y1, x0:x1]
                band = cv2.resize(band, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
                gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                raw_txt = pytesseract.image_to_string(gray, config="--psm 6").strip()
                candidate = clean_ocr_name(raw_txt)
                if candidate:
                    label_samples[tile_idx].append(candidate)

            if debug_snapshots and (processed % max(1, snapshot_every) == 0):
                try:
                    cv2.imwrite(str((temp_dir / f"debug_{R}x{C}_{snap:04d}.jpg")), frame_c); snap += 1
                except Exception: pass

            f_idx += 1; processed += 1
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

    # ---- after trying all grids, we pick identities using the chosen grid ----
    identities: Dict[int, TileIdentity] = {}
    R, C = best_grid
    for idx in range(R*C):
        samples = best_labels.get(idx, [])
        if not samples:
            raw = ""
        else:
            vals, cnts = np.unique(np.array(samples), return_counts=True)
            raw = str(vals[int(np.argmax(cnts))])
        identities[idx] = TileIdentity(idx, raw, "")
    cap.release()
    log(f"[Visual] Selected grid: {R}x{C} (stability={best_score:.3f})")
    return best_events, identities, (R,C)

def first_stable_highlight_time(events: List[TileEvent], stable_len: int = 3) -> Optional[float]:
    if not events: return None
    for i in range(len(events)-stable_len+1):
        tid = events[i].tile_idx
        if all(events[i+j].tile_idx == tid for j in range(1, stable_len)): return events[i].t
    return events[0].t

def refine_offset_grid(events: List[TileEvent], stt: List[STTSegment], initial_offset: float,
                       search_window: float=10.0, step: float=0.1) -> float:
    change_times = [events[i].t for i in range(1,len(events)) if events[i].tile_idx != events[i-1].tile_idx]
    seg_starts = [s.start for s in stt]
    if not change_times or not seg_starts: return initial_offset

    def bin_times(times: List[float]) -> np.ndarray:
        if not times: return np.zeros(1, dtype=int)
        tmax = max(times)+1.0; n=int(math.ceil(tmax/0.25))
        arr = np.zeros(n, dtype=int)
        for t in times:
            idx = min(n-1, int(round(t/0.25))); arr[idx]=1
        return arr

    vb = bin_times(change_times); best_off=initial_offset; best_score=-1.0
    for off in np.arange(initial_offset - search_window, initial_offset + search_window + 1e-9, step):
        sb = bin_times([t+off for t in seg_starts])
        M = max(len(vb), len(sb))
        vb2 = np.pad(vb, (0, M-len(vb))); sb2 = np.pad(sb, (0, M-len(sb)))
        inter = np.sum((vb2==1) & (sb2==1)); union = np.sum((vb2==1) | (sb2==1)) + 1e-6
        score = inter/union
        if score > best_score: best_score=score; best_off=float(off)
    log(f"[Align] initial={initial_offset:.2f}s refined={best_off:.2f}s score={best_score:.3f}")
    return best_off

def map_identities_to_ics(identities: Dict[int, TileIdentity], attendees: List[Attendee]) -> Dict[int, TileIdentity]:
    for idx, ident in identities.items():
        raw = (ident.label_raw or "").strip()
        if not _is_plausible_person_name(raw):
            identities[idx] = TileIdentity(idx, raw, "")
            continue
        mapped = strict_map_to_ics(raw, attendees, cutoff=88)
        identities[idx] = TileIdentity(idx, raw, mapped)
    return identities

def merge_consecutive_segments(attributed: List[Tuple[str,float,float,str]]) -> List[Tuple[str,float,float,str]]:
    if not attributed: return []
    m=[attributed[0]]
    for name,s,e,txt in attributed[1:]:
        pn,ps,pe,pt = m[-1]
        if name==pn and abs(s-pe)<0.75: m[-1]=(pn,ps,e,(pt+" "+txt).strip())
        else: m.append((name,s,e,txt))
    return m

def format_transcript_lines(attributed: List[Tuple[str,float,float,str]]) -> List[str]:
    return [f"{(n or 'Unknown')}: {t}" for n,s,e,t in attributed]

def ensure_work_dirs(base: Path):
    out_dir = base / "out"; temp_dir = base / "temp"
    out_dir.mkdir(parents=True, exist_ok=True); temp_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, temp_dir

def pick_inputs_with_gui(cfg: dict) -> dict:
    import tkinter as tk
    from tkinter import filedialog
    root=tk.Tk(); root.withdraw()
    if cfg["VISUAL_VIDEO"] is None:
        cfg["VISUAL_VIDEO"] = Path(filedialog.askopenfilename(title="Select Zoom screen recording (visual only)",
            filetypes=[("Video","*.mp4 *.mov *.mkv *.avi *.webm")]))
    if cfg["AUDIO_SOURCE"] is None:
        cfg["AUDIO_SOURCE"] = Path(filedialog.askopenfilename(title="Select audio source (video w/ audio OR audio)",
            filetypes=[("Media","*.mp4 *.mov *.mkv *.avi *.webm *.wav *.m4a *.mp3 *.aac *.flac *.ogg *.opus")]))
    if cfg["ICS_FILE"] is None:
        cfg["ICS_FILE"] = Path(filedialog.askopenfilename(title="Select .ics invite",
            filetypes=[("iCalendar","*.ics")]))
    if cfg["WORK_DIR"] is None:
        cfg["WORK_DIR"] = Path(filedialog.askdirectory(title="Select Working Folder (will create out/ and temp/)"))
        # Trick: if you prefer a directory chooser:
        # cfg["WORK_DIR"] = Path(filedialog.askdirectory(title="Select Working Folder"))
        # but keep the above if your environment blocks askdirectory dialogs
    root.destroy(); return cfg

def write_outputs(vis_path: Path, aud_src: Path, ics_path: Optional[Path],
                  out_dir: Path, temp_dir: Path, lines: List[str],
                  attendees: List[Attendee], identities: Dict[int,TileIdentity],
                  best_offset: float, stt_segments_count: int,
                  grid_info: Optional[Tuple[int,int]], status: str):
    base = vis_path.stem + "_fused"
    out_txt = out_dir / f"{base}_attributed_transcript.txt"
    out_json = out_dir / f"{base}_diagnostics.json"
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    diag = {
        "status": status,
        "visual_video": str(vis_path),
        "audio_source": str(aud_src),
        "ics": str(ics_path) if ics_path else "",
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

# ---- atexit emergency writer ----
_emergency = {"enabled": False, "vis_path": None, "aud_src": None, "ics_path": None,
              "out_dir": None, "temp_dir": None, "attendees": [], "stt_segments": None}

def _atexit_emergency_writer():
    try:
        if not _emergency["enabled"]:
            return
        vis_path = _emergency["vis_path"]
        out_dir  = _emergency["out_dir"]
        temp_dir = _emergency["temp_dir"]
        aud_src  = _emergency["aud_src"]
        ics_path = _emergency["ics_path"]
        attendees = _emergency["attendees"]
        stt_segments = _emergency["stt_segments"]
        if not (vis_path and out_dir and stt_segments):
            return

        # if outputs already present, do nothing
        base = vis_path.stem + "_fused"
        diag_path = out_dir / f"{base}_diagnostics.json"
        if diag_path.exists():
            try:
                st = json.loads(diag_path.read_text(encoding="utf-8")).get("status", "")
                if st in {"success", "stt_only_provisional"}:
                    log("[atexit] Outputs already present; skipping emergency write.")
                    return
            except Exception:
                pass

        # otherwise write STT-only fallback
        lines = [f"Unknown: {s.text}" for s in stt_segments]
        write_outputs(vis_path, aud_src, ics_path, out_dir, temp_dir,
                      lines, attendees, {}, 0.0, len(stt_segments), None, "stt_only")
        log("[atexit] Wrote emergency STT-only outputs.")
    except Exception as ex:
        elog("[atexit] Emergency write failed: " + str(ex))

atexit.register(_atexit_emergency_writer)

def ensure_audio_ready(aud_src: Path, temp_dir: Path) -> Path:
    if is_mono_16k_wav(aud_src):
        target = temp_dir / aud_src.name
        if str(target).lower() != str(aud_src).lower():
            try:
                target.write_bytes(aud_src.read_bytes())
                log(f"[Audio] Using existing mono/16k WAV (copied to temp): {target}")
            except Exception:
                target = aud_src; log(f"[Audio] Using existing mono/16k WAV (original): {target}")
        else:
            log(f"[Audio] Using existing mono/16k WAV: {target}")
        return target
    out_wav = temp_dir / (aud_src.stem + "_audio16k.wav")
    log(f"[Audio] Extracting/normalizing to WAV: {out_wav}")
    extract_audio_to_wav(aud_src, out_wav, 16000)
    log(f"[Audio] WAV ready")
    return out_wav

# NEW: helper to save a grid preview image with tile indices
def _save_grid_preview(video_path: Path, grid: Tuple[int,int], out_path: Path, sample_time_s: Optional[float] = None) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if sample_time_s is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(sample_time_s))*1000.0)
    else:
        # fallback: 20% into the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames*0.2))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return
    frame_c, _off = _crop_canvas(frame, CONFIG.get("CANVAS_CROP", {}))
    R, C = grid
    h, w = frame_c.shape[:2]
    th = h // R; tw = w // C
    for r in range(R):
        for c in range(C):
            y0, y1 = r*th, min((r+1)*th, h)
            x0, x1 = c*tw, min((c+1)*tw, w)
            idx = r*C + c
            cv2.rectangle(frame_c, (x0, y0), (x1-1, y1-1), (255, 255, 255), 2)
            cv2.putText(frame_c, f"Tile{idx+1} ({idx})", (x0+10, y0+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), frame_c)

# NEW: helper to write a tile-activity CSV (approx “talk time” per tile)
def _write_tile_activity_csv(events: List["TileEvent"], grid: Tuple[int,int], out_csv: Path) -> None:
    if not events:
        return
    totals: Dict[int, float] = {}
    for i in range(len(events)-1):
        t0, t1 = events[i].t, events[i+1].t
        idx = events[i].tile_idx
        totals[idx] = totals.get(idx, 0.0) + max(0.0, t1 - t0)
    rows = sorted(((idx, totals.get(idx, 0.0)) for idx in range(grid[0]*grid[1])),
                  key=lambda x: x[1], reverse=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tile_idx", "approx_seconds"])
        w.writerows(rows)

def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False); parser.add_argument("--nogui", action="store_true")
    args,_ = parser.parse_known_args(argv)

    cfg = dict(CONFIG)
    if USE_GUI_DEFAULT and not args.nogui:
        cfg = pick_inputs_with_gui(cfg)

    vis_path = Path(cfg["VISUAL_VIDEO"]) if cfg["VISUAL_VIDEO"] else None
    aud_src  = Path(cfg["AUDIO_SOURCE"]) if cfg["AUDIO_SOURCE"] else None
    ics_path = Path(cfg["ICS_FILE"]) if cfg["ICS_FILE"] else None
    work_dir = Path(cfg["WORK_DIR"]) if cfg["WORK_DIR"] else None

    print("[FILES] Visual:", vis_path or "<none>")
    print("[FILES] Audio: ", aud_src or "<none>")
    print("[FILES] ICS:   ", ics_path or "<none>")

    if not vis_path or not vis_path.exists(): raise FileNotFoundError("Missing VISUAL_VIDEO")
    if not aud_src or not aud_src.exists():  raise FileNotFoundError("Missing AUDIO_SOURCE")
    if not work_dir:                         raise FileNotFoundError("Missing WORK_DIR")

    out_dir, temp_dir = ensure_work_dirs(work_dir)
    run_log_path = out_dir / (vis_path.stem + "_run.log"); _open_log_file(run_log_path)

    log(f"[Paths] Work: {work_dir}")
    log(f"[Paths] Out:  {out_dir}")
    log(f"[Paths] Temp: {temp_dir}")

    attendees = parse_ics_attendees(ics_path) if (ics_path and ics_path.exists()) else []
    log(f"[ICS] Loaded attendees: {len(attendees)}")

    wav_path = ensure_audio_ready(aud_src, temp_dir)

    # ---- STT ----
    stt_segments = transcribe_wav_fwhisper(wav_path, cfg["WHISPER_MODEL"], cfg["WHISPER_DEVICE"], cfg["WHISPER_COMPUTE"])
    if not stt_segments:
        log("[STT] No segments — writing empty transcript.")
        write_outputs(vis_path, aud_src, ics_path, out_dir, temp_dir, [], attendees, {}, 0.0, 0, None, "stt_only")
        close_log(); return 2

    # enable atexit emergency now that we have STT
    _emergency.update({"enabled": True, "vis_path": vis_path, "aud_src": aud_src, "ics_path": ics_path,
                       "out_dir": out_dir, "temp_dir": temp_dir, "attendees": attendees, "stt_segments": stt_segments})

    # Provisional write
    provisional_lines = [f"Unknown: {s.text}" for s in stt_segments]
    write_outputs(vis_path, aud_src, ics_path, out_dir, temp_dir, provisional_lines, attendees, {}, 0.0, len(stt_segments), None, "stt_only_provisional")
    log("[Checkpoint] Provisional transcript+diagnostics written. Proceeding to visual...")

    # ---- VISUAL PREFLIGHT ----
    log("[Visual] Preflight: opening video...")
    cap0 = _open_video_with_backoffs(vis_path)
    if cap0 is None or not cap0.isOpened():
        log(f"[Visual][ERROR] Cannot open video: {vis_path}")
        close_log(); return 4
    ok, first = cap0.read(); cap0.release()
    if not ok or first is None:
        log(f"[Visual][ERROR] First frame read failed: {vis_path}")
        close_log(); return 5
    log("[Visual] Preflight OK; starting detection...")

    # ---- VISUAL DETECTION ----
    try:
        events, identities, grid = detect_highlight_series(
            vis_path,
            fps_sample=float(cfg["FPS_SAMPLE"]),
            do_ocr=bool(cfg["OCR_ENABLED"] and pytesseract is not None),
            debug_snapshots=bool(cfg["DEBUG_SNAPSHOTS"]),
            snapshot_every=int(cfg["SNAPSHOT_EVERY"]),
            temp_dir=temp_dir
        )
    except Exception as ex:
        elog("[Visual][FATAL] " + str(ex)); traceback.print_exc()
        close_log(); return 6

    if not events:
        log("[Visual] No highlight events — keeping STT-only.")
        close_log(); return 3

    R, C = grid
    log(f"[Visual] Events: {len(events)}, Grid: {R}x{C}")

    # grid preview + tile activity aids
    base = vis_path.stem + "_fused"
    preview_path = out_dir / f"{base}_grid_preview.jpg"
    mid_t = events[len(events)//2].t if events else 0.0
    _save_grid_preview(vis_path, grid, preview_path, sample_time_s=mid_t)
    log(f"[OUT] Grid preview: {preview_path}")

    activity_csv = out_dir / f"{base}_tile_activity.csv"
    _write_tile_activity_csv(events, grid, activity_csv)
    log(f"[OUT] Tile activity: {activity_csv}")

    identities = map_identities_to_ics(identities, attendees)

    t_vis0 = first_stable_highlight_time(events) or 0.0
    t_aud0 = stt_segments[0].start
    initial = t_vis0 - t_aud0
    log(f"[Align] Initial offset = {initial:.2f}s")
    best_offset = refine_offset_grid(events, stt_segments, initial, search_window=10.0, step=0.1)

    times_arr = np.array([ev.t for ev in events], dtype=np.float32)
    tiles_arr = np.array([ev.tile_idx for ev in events], dtype=np.int16)

    # Attribution (apply overrides first)
    overrides = cfg.get("TILE_NAME_OVERRIDES", {}) or {}
    attributed: List[Tuple[str,float,float,str]] = []
    for seg in stt_segments:
        t_vis = seg.start + best_offset
        idx = int(np.clip(np.searchsorted(times_arr, t_vis, side="right") - 1, 0, len(times_arr)-1))
        tidx = int(tiles_arr[idx]); ident = identities.get(tidx)
        name = overrides.get(tidx) or ((ident.name_mapped or ident.label_raw or f"Tile{tidx+1}") if ident else f"Tile{tidx+1}")
        attributed.append((name, seg.start, seg.end, seg.text))

    attributed = merge_consecutive_segments(attributed)
    final_lines = format_transcript_lines(attributed)
    write_outputs(vis_path, aud_src, ics_path, out_dir, temp_dir, final_lines, attendees, identities, best_offset, len(stt_segments), (R,C), "success")
    log(f"[DONE] Attributed lines: {len(final_lines)}")

    # prevent clobbering success files on process exit
    _emergency["enabled"] = False
    log("[Success] Final outputs written; emergency atexit disabled.")
    # Optional: release model only after success writes
    # global _WHISPER_MODEL; _WHISPER_MODEL = None; gc.collect()

    close_log()
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as ex:
        elog("[FATAL] " + str(ex)); traceback.print_exc(); close_log(); sys.exit(1)
