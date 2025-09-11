#!/usr/bin/env python3
"""
Attributor V3.1
================

Purpose
-------
Roll-up of V3 with fixes for your curved-screen capture + dynamic layouts:

1) Perspective pre-warp (optional):
   - CONFIG["WARP_QUAD"] = four (x,y) points on the video that bracket the Teams canvas
   - CONFIG["WARP_SIZE"] = (width, height) of the rectified output (e.g., (1280,720))
   - Each video frame is rectified before detection/scoring.

2) Dynamic layout estimator from nameplates (robust to participants joining/leaving):
   - Samples the bottom-band OCR across the *video frame* every N frames
   - Infers column/row centers via 1-D k-means (k chosen automatically in [2..DYNAMIC_MAX_SIDE])
   - Converts centers → grid bounds and slices tiles accordingly (no dependency on an external screenshot).

3) Safer defaults for screenshot OCR (when used only for names):
   - Avoids left-shifted grid by disabling token-based L/R and clamping width to ~full width
   - Slightly taller bottom label band; smaller inset

4) Keep V3 scoring: sums blue-mask pixels *inside each tile* (resolution-agnostic)
   - Optional hue relax for pastel borders (--relaxed-blue-threshold)

5) Optional convenience flag: --tiles-from-video
   - Grabs a representative frame from the video and uses it for OCR-to-names.

Run examples
------------
Dynamic + warp + relaxed blue:
    python attributor_v3_1.py \
        --video meeting.mp4 \
        --ics invite.ics \
        --outdir out \
        --dynamic-grid \
        --relaxed-blue-threshold 0.35

If you want names from a frame in the video (no external screenshot):
    python attributor_v3_1.py \
        --video meeting.mp4 --tiles-from-video \
        --ics invite.ics --outdir out --dynamic-grid --relaxed-blue-threshold 0.35

To set a perspective warp, edit CONFIG["WARP_QUAD"] and CONFIG["WARP_SIZE"].
Pick the *inner* corners of the Teams canvas in the recorded video.

"""

from __future__ import annotations
import argparse, dataclasses, json, os, re, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ============================== CONFIG ==============================
CONFIG = {
    # STT
    "LANGUAGE": "en",
    "PREFER_FASTER": True,
    "STT_MODEL": "medium",
    "DEVICE": "cuda",
    "COMPUTE_TYPE": "float16",
    "CPU_THREADS": 0,
    "WHISPER_WORKERS": 2,

    # Video sampling
    "FPS_SAMPLE": 3.0,
    "VIDEO_WORKERS": 4,

    # Blue border HSV (Teams/Zoom typical blue)
    "LOWER_HSV": (85, 50, 50),
    "UPPER_HSV": (140, 255, 255),

    # Resolution-agnostic area floor for blue edges (fraction of full-frame area)
    "MIN_EDGE_AREA_FRAC": 0.0005,   # ~0.05% of frame

    # Optional hue relax factor (None to disable; e.g., 0.35 widens hue band)
    "RELAXED_BLUE_THRESHOLD": None,

    # Optional crop (x, y, w, h) to limit detection region (leave None for full frame)
    "VIDEO_CROP": None,

    # Optional perspective warp (handles curved-screen camera capture)
    # Four points (x,y) in *video* coordinates ordered TL, TR, BR, BL.
    "WARP_QUAD": None,  # e.g., [(130,120),(1140,120),(1140,650),(130,650)]
    "WARP_SIZE": None,  # e.g., (1280, 720)

    # OCR
    "TESSERACT_CMD": None,
    "OCR_LANG": "eng",
    "OCR_ALLOWED_RE": r"[^A-Za-z0-9 .,'()@\-\[\]]",
    "OCR_MIN_LEN": 2,
    "FUZZ_MIN_SCORE": 55,
    "OCR_WORD_CONF_MIN": 40,
    "INITIALS_CONF_MIN": 50,

    # Screenshot OCR geometry — safer defaults to avoid left-shift
    "AUTO_LR_FROM_TOKENS": False,
    "LR_MARGIN_FRAC": 0.00,
    "MIN_GRID_WIDTH_FRAC": 0.95,
    "AUTO_TOP_OFFSET": True,
    "TOP_OFFSET_FRAC": 0.09,
    "AUTO_TOP_MIN_FRAC": 0.10,
    "AUTO_TOP_MAX_FRAC": 0.20,
    "AUTO_TOP_EXTRA_FRAC": 0.03,
    "BOTTOM_STRIP_FRAC": 0.55,
    "BOTTOM_STRIP_PAD_BOTTOM_FRAC": 0.010,
    "GRID_INSET": 3,
    "LARGE_BUBBLE_AREA_FRAC": 0.02,

    # Dynamic grid
    "DYNAMIC_GRID": False,
    "DYNAMIC_MAX_SIDE": 4,
    "ONFRAME_OCR": True,
    "TRANSCRIPT_ONLY_PLACEHOLDER": "UnknownSpeaker",
    "FALLBACK_IF_NO_SCREENSHOT": True,
    "EXPORT_DYNAMIC_MAP": True,
    "DYNAMIC_NAME_CACHE_TTL": 150,

    # Dynamic layout estimator
    "LAYOUT_UPDATE_PERIOD": 10,   # frames between layout refreshes
    "LAYOUT_USE_BAND": (0.60, 0.92),  # bottom band (as frac of H) to OCR for nameplates

    # Debug
    "DEBUG": True,
    "DEBUG_EVERY_N": 15,
}

# ============================ Imports ============================
import numpy as np
import cv2
from PIL import Image  # noqa: F401
import pytesseract
from icalendar import Calendar
from rapidfuzz import fuzz, process as rf_process

# ----------------------------- STT Engines -----------------------------
class STTEngine:
    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> List[Dict]:
        raise NotImplementedError

class FasterWhisperEngine(STTEngine):
    def __init__(self, model_size: str, device: str, compute_type: str, cpu_threads: int, num_workers: int):
        print("[STT] Loading faster-whisper...", flush=True)
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_size,
                                  device=device or "auto",
                                  compute_type=compute_type,
                                  cpu_threads=cpu_threads,
                                  num_workers=num_workers)
        self.device = device
        self.compute_type = compute_type

    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> List[Dict]:
        kwargs = {"vad_filter": True}
        if language:
            kwargs["language"] = language
        segments, _ = self.model.transcribe(str(audio_path), **kwargs)
        out = []
        for seg in segments:
            out.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
                "prob": float(getattr(seg, "avg_logprob", 0.0)),
            })
        return out

class OpenAIWhisperEngine(STTEngine):
    def __init__(self, model_size: str):
        print("[STT] Loading openai-whisper...", flush=True)
        import whisper, torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size, device=device)
        self.device = device

    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> List[Dict]:
        import whisper
        kwargs = {"verbose": False}
        if language:
            kwargs["language"] = language
        result = self.model.transcribe(str(audio_path), **kwargs)
        out = []
        for seg in result.get("segments", []):
            out.append({
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg["text"].strip(),
                "prob": float(seg.get("avg_logprob", 0.0)),
            })
        return out

def safe_transcribe(audio: Path, language: Optional[str]):
    used = {}
    try:
        fw = FasterWhisperEngine(CONFIG["STT_MODEL"], CONFIG["DEVICE"], CONFIG["COMPUTE_TYPE"],
                                 CONFIG["CPU_THREADS"], CONFIG["WHISPER_WORKERS"])
        print(f"[STT] Trying faster-whisper on {fw.device} ({fw.compute_type}) ...")
        segs = fw.transcribe(audio, language=language)
        used = {"engine": "faster-whisper", "device": fw.device, "compute_type": fw.compute_type}
        return segs, used
    except Exception as e:
        print(f"[STT] faster-whisper GPU failed: {e}")
    try:
        fw_cpu = FasterWhisperEngine(CONFIG["STT_MODEL"], "cpu", "int8", CONFIG["CPU_THREADS"], CONFIG["WHISPER_WORKERS"])
        print("[STT] Retrying faster-whisper on CPU (int8) ...")
        segs = fw_cpu.transcribe(audio, language=language)
        used = {"engine": "faster-whisper", "device": "cpu", "compute_type": "int8"}
        return segs, used
    except Exception as e:
        print(f"[STT] faster-whisper CPU failed: {e}")
    try:
        ow = OpenAIWhisperEngine(CONFIG["STT_MODEL"])
        print(f"[STT] Falling back to openai-whisper on {ow.device} ...")
        segs = ow.transcribe(audio, language=language)
        used = {"engine": "openai-whisper", "device": ow.device}
        return segs, used
    except Exception as e:
        raise RuntimeError(f"No STT engine could run: {e}")

# ----------------------------- Data Models -----------------------------
@dataclass
class Tile:
    name: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    grid_pos: Tuple[int, int]        # row, col

@dataclass
class SpeakerEvent:
    start: float
    end: float
    tile_name: str
    validated: bool

@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    validated: bool = False
    prob: Optional[float] = None

# ----------------------------- Utilities -----------------------------
def run_ffmpeg_extract_audio(video: Path, out_wav: Path, sample_rate: int = 16000) -> None:
    cmd = ["ffmpeg","-y","-i",str(video),"-ac","1","-ar",str(sample_rate),str(out_wav)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', errors='ignore')}")

def hhmmss(seconds: float) -> str:
    total = int(max(0, seconds))
    ms = int((max(0.0, seconds) - int(max(0, seconds))) * 1000)
    h = total // 3600; m = (total % 3600) // 60; s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

# ----------------------------- ICS Parsing (extended) -----------------------------
from icalendar import Calendar

def parse_ics_attendees_extended(ics_path: Path) -> Tuple[Dict[str,str], Dict[str,str], Dict[str,List[str]]]:
    text = ics_path.read_bytes()
    cal = Calendar.from_ical(text)

    def domain_to_company(email: str) -> str:
        dom = email.split("@")[-1].split(">")[0].strip().lower()
        base = dom.split(".")[0] if "." in dom else dom
        mapping = {"omitron": "Omitron"}
        return mapping.get(base, base.capitalize())

    name_to_company: Dict[str,str] = {}
    email_to_name: Dict[str,str] = {}
    all_names: List[str] = []

    def record(name: Optional[str], email: Optional[str]):
        if not name and not email: return
        nm = name.strip() if name else None
        em = email.strip().lower() if email else None
        comp = domain_to_company(em) if em else ""
        if nm:
            name_to_company[nm] = name_to_company.get(nm, comp) or comp
            if nm not in all_names: all_names.append(nm)
        if em and nm:
            email_to_name[em] = nm

    for comp in cal.walk():
        if comp.name != "VEVENT": continue
        org = comp.get("ORGANIZER") or comp.get("organizer")
        if org:
            sval = str(org)
            mcn = re.search(r"CN=([^:;]+)", sval); men = re.search(r"mailto:([^>\s]+)", sval, re.I)
            record(mcn.group(1) if mcn else None, men.group(1) if men else None)
        raw_atts = comp.get("attendee") or comp.get("ATTENDEE") or []
        if not isinstance(raw_atts, list): raw_atts = [raw_atts]
        for att in raw_atts:
            try:
                params = getattr(att, "params", {})
            except Exception:
                params = {}
            cn = None
            if params and "CN" in params: cn = str(params["CN"]).strip()
            sval = str(att)
            if not cn:
                m = re.search(r"CN=([^:;]+)", sval)
                cn = m.group(1).strip() if m else None
            men = re.search(r"mailto:([^>\s]+)", sval, re.I)
            record(cn, men.group(1) if men else None)

    # initials map
    from collections import defaultdict
    initials_map: Dict[str, List[str]] = defaultdict(list)
    for nm in all_names:
        parts = [p for p in re.split(r"[\s,]+", nm) if p]
        if len(parts)>=2:
            init = (parts[0][0] + parts[-1][0]).upper()
        else:
            init = parts[0][0].upper() if parts else ""
        if init: initials_map[init].append(nm)

    return name_to_company, email_to_name, dict(initials_map)

# ----------------------------- OCR helpers -----------------------------
def set_tesseract_cmd_if_provided():
    if CONFIG["TESSERACT_CMD"]:
        pytesseract.pytesseract.tesseract_cmd = CONFIG["TESSERACT_CMD"]

def _clean_text(txt: str) -> str:
    txt = re.sub(r"[\r\n]+", " ", txt).strip()
    txt = re.sub(CONFIG["OCR_ALLOWED_RE"], "", txt)
    txt = re.sub(r"\s{2,}", " ", txt)
    return txt.strip()

def _kmeans1d(vals: List[float], k: int = 3, iters: int = 30) -> Optional[List[float]]:
    if not vals or len(vals) < k:
        return None
    v = np.array(sorted(vals), dtype=np.float32)
    qs = np.linspace(0, 1, k + 2)[1:-1]
    cent = np.array([np.quantile(v, q) for q in qs], dtype=np.float32)
    for _ in range(iters):
        d = np.abs(v[:, None] - cent[None, :])
        lab = np.argmin(d, axis=1)
        new = np.array([(v[lab == i].mean() if np.any(lab == i) else cent[i]) for i in range(k)], dtype=np.float32)
        if np.allclose(new, cent, atol=1e-2):
            break
        cent = new
    cent.sort()
    return cent.tolist()

def _bounds_from_centers_generic(cs: List[float], lo: int, hi: int) -> List[int]:
    """Given sorted centers, return list of bounds of length len(cs)+1 clamped to [lo,hi]."""
    if not cs:
        return [lo, hi]
    cs = sorted(cs)
    bounds = [lo]
    for i in range(1, len(cs)):
        bounds.append(int(round((cs[i-1] + cs[i]) / 2.0)))
    bounds.append(hi)
    # clamp monotonic
    for i in range(1, len(bounds)):
        if bounds[i] <= bounds[i-1]:
            bounds[i] = bounds[i-1] + 1
    bounds[0] = max(bounds[0], lo)
    bounds[-1] = min(bounds[-1], hi)
    return bounds

def _preproc_gray(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    return gray

def _tess_data(img, psm: int) -> List[Dict[str, Any]]:
    custom = f'--oem 3 --psm {psm} -l {CONFIG["OCR_LANG"]}'
    d = pytesseract.image_to_data(img, config=custom, output_type=pytesseract.Output.DICT)
    out = []
    n = len(d.get("text", []))
    for i in range(n):
        txt = d["text"][i]
        if not txt: continue
        try: conf = float(d.get("conf", ["-1"])[i])
        except: conf = -1.0
        x = int(d["left"][i]); y = int(d["top"][i]); w = int(d["width"][i]); h = int(d["height"][i])
        out.append({"text": _clean_text(txt), "conf": conf, "bbox": (x,y,w,h)})
    return out

def _auto_top_offset(gray: np.ndarray) -> int:
    if not CONFIG["AUTO_TOP_OFFSET"]:
        return int(CONFIG["TOP_OFFSET_FRAC"] * gray.shape[0])
    H = gray.shape[0]
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    row_energy = mag.mean(axis=1)
    search_to = int(0.25 * H)
    if search_to < 10:
        return int(CONFIG["TOP_OFFSET_FRAC"] * H)
    k = max(3, (int(0.01 * H) | 1))
    row_energy_s = cv2.GaussianBlur(row_energy.astype(np.float32), (k, 1), 0)
    idx = int(np.argmax(row_energy_s[:search_to]))
    extra = int(CONFIG["AUTO_TOP_EXTRA_FRAC"] * H)
    lo = int(CONFIG["AUTO_TOP_MIN_FRAC"] * H)
    hi = int(CONFIG["AUTO_TOP_MAX_FRAC"] * H)
    off = max(lo, min(hi, idx + extra))
    return off

def _union_bbox(b1, b2):
    x1,y1,w1,h1 = b1; x2,y2,w2,h2 = b2
    xa = min(x1,x2); ya = min(y1,y2)
    xb = max(x1+w1, x2+w2); yb = max(y1+h1, y2+h2)
    return (xa, ya, xb-xa, yb-ya)

def _strip_badges(t: str) -> str:
    t = re.sub(r"\[[^\]]+\]", " ", t)
    t = re.sub(r"\([^\)]+\)", " ", t)
    return re.sub(r"\s{2,}", " ", t).strip()

def _normalize_label_for_match(text: str) -> str:
    t = re.sub(r"\[[^\]]+\]|\([^)]+\)", " ", text)
    t = re.sub(r"\s{2,}", " ", t).strip()
    m = re.match(r"^([A-Za-z.'-]+),\s*([A-Za-z.'-]+)$", t)
    if m:
        t = f"{m.group(2)} {m.group(1)}"
    return t

# ----------------------------- Screenshot OCR → tiles (names) -----------------------------
def ocr_to_tiles(screenshot_path: Path, attendees_name_to_company: Dict[str,str],
                 email_to_name: Dict[str,str], initials_map: Dict[str,List[str]], outdir: Path):
    set_tesseract_cmd_if_provided()
    img = cv2.imread(str(screenshot_path))
    if img is None:
        raise RuntimeError(f"Failed to read screenshot: {screenshot_path}")

    H, W = img.shape[:2]
    gray = _preproc_gray(img)
    top_off = _auto_top_offset(gray)
    grid_h = max(1, H - top_off)
    tile_h = int(grid_h / 3.0)

    # Pass A: OCR grid-only
    roi_full = gray[top_off:H, :]
    wordsA = _tess_data(roi_full, psm=6)
    for w in wordsA:
        x, y, wid, hei = w["bbox"]
        w["bbox"] = (x, top_off + y, wid, hei)

    # Pass B: bottom strips (single-line)
    pad_b = float(CONFIG.get("BOTTOM_STRIP_PAD_BOTTOM_FRAC", 0.010))
    bs_frac = float(CONFIG.get("BOTTOM_STRIP_FRAC", 0.55))
    wordsB = []
    for r in range(3):
        y1 = top_off + r * tile_h + int(tile_h * bs_frac)
        y2 = top_off + (r + 1) * tile_h - int(tile_h * pad_b)
        y1 = max(top_off, min(H - 2, y1))
        y2 = max(y1 + 4, min(H, y2))
        roi = gray[y1:y2, :]
        if roi.size == 0:
            continue
        data = _tess_data(roi, psm=7)
        for w in data:
            x, y, wid, hei = w["bbox"]
            w["bbox"] = (x, y1 + y, wid, hei)
        wordsB.extend(data)

    # Pass C: initials in centers
    wordsC = []
    tile_w_tmp = int(W / 3.0)
    for r in range(3):
        for c in range(3):
            cx1 = int(c * tile_w_tmp + 0.25 * tile_w_tmp)
            cy1 = int(top_off + r * tile_h + 0.25 * tile_h)
            cx2 = int((c + 1) * tile_w_tmp - 0.25 * tile_w_tmp)
            cy2 = int(top_off + (r + 1) * tile_h - 0.25 * tile_h)
            cx1 = max(0, min(W-2, cx1)); cx2 = max(cx1+4, min(W, cx2))
            cy1 = max(top_off, min(H-2, cy1)); cy2 = max(cy1+4, min(H, cy2))
            roi = gray[cy1:cy2, cx1:cx2]
            if roi.size == 0: continue
            data = _tess_data(roi, psm=8)
            for w in data:
                x, y, wid, hei = w["bbox"]
                w["bbox"] = (cx1 + x, cy1 + y, wid, hei)
            wordsC.extend(data)

    words_all = wordsA + wordsB + wordsC
    words_keep = [w for w in words_all if w["conf"] >= CONFIG["OCR_WORD_CONF_MIN"] and _clean_text(w["text"])]
    grid_min_y = top_off + 0.05 * tile_h
    def _cy(bb): x,y,w,h = bb; return y + h/2.0
    words_keep = [w for w in words_keep if _cy(w["bbox"]) >= grid_min_y]

    # Use full-width thirds to avoid left-shift (AUTO_LR_FROM_TOKENS disabled)
    col_bounds = [0, W//3, 2*W//3, W]
    row_bounds = [top_off, top_off + tile_h, top_off + 2*tile_h, H]
    inset = int(CONFIG.get("GRID_INSET", 3))

    def _which_bin(val: float, edges: List[int]) -> int:
        # edges length 4 → returns 0..2
        if val < edges[1]: return 0
        if val < edges[2]: return 1
        return 2

    def _tile_bbox_from_point_grid(cx, cy):
        col = _which_bin(cx, col_bounds)
        row = _which_bin(cy, row_bounds)
        x1, x2 = col_bounds[col], col_bounds[col + 1]
        y1, y2 = row_bounds[row], row_bounds[row + 1]
        bx, by = x1 + inset, y1 + inset
        bw, bh = max(1, (x2 - x1) - 2*inset), max(1, (y2 - y1) - 2*inset)
        return (bx, by, bw, bh), (row, col)

    # Initials-first + large-bubble heuristic
    attendee_names = list(attendees_name_to_company.keys())
    kept_tiles: Dict[str, Tuple[int,int,int,int]] = {}
    kept_score: Dict[str, float] = {}

    for w in words_keep:
        t = _clean_text(w["text"]).upper()
        if re.fullmatch(r"[A-Z]{2}", t):
            tile_bb, _ = _tile_bbox_from_point_grid(w["bbox"][0] + w["bbox"][2]/2.0,
                                                    w["bbox"][1] + w["bbox"][3]/2.0)
            cell_area = tile_bb[2] * tile_bb[3]
            token_area = w["bbox"][2] * w["bbox"][3]
            large_bubble = token_area >= CONFIG["LARGE_BUBBLE_AREA_FRAC"] * cell_area
            conf_ok = w["conf"] >= CONFIG["INITIALS_CONF_MIN"]
            if conf_ok or large_bubble:
                names = initials_map.get(t, [])
                if len(names) == 1:
                    name = names[0]
                    score = max(w["conf"], 80.0 if large_bubble else w["conf"])
                    if score > kept_score.get(name, -1):
                        kept_tiles[name] = tile_bb
                        kept_score[name] = score

    # Phrase assembly (bottom strips)
    cell_words = [w for w in words_keep if (w["bbox"][1] + w["bbox"][3]/2.0) >= top_off + tile_h * bs_frac]
    # define bs_frac in local scope (same as used above)
    # (re-declare to avoid UnboundLocal if edited)
    bs_frac = float(CONFIG.get("BOTTOM_STRIP_FRAC", 0.55))
    cell_words = [w for w in words_keep if (w["bbox"][1] + w["bbox"][3]/2.0) >= top_off + tile_h * bs_frac]
    cell_words.sort(key=lambda z: (z["bbox"][1], z["bbox"][0]))

    lines: List[List[Dict[str, Any]]] = []
    for w in cell_words:
        if not lines: lines.append([w]); continue
        last_y = sum([ww["bbox"][1] for ww in lines[-1]]) / len(lines[-1])
        if abs(w["bbox"][1] - last_y) <= 22: lines[-1].append(w)
        else: lines.append([w])

    def union_bbox_chain(toks):
        bb = toks[0]["bbox"]
        for t in toks[1:]:
            bb = _union_bbox(bb, t["bbox"])
        return bb

    for line in lines:
        line.sort(key=lambda z: z["bbox"][0])
        n = len(line)
        for i in range(n):
            for L in range(1, 8):
                if i + L > n: break
                toks = line[i:i+L]
                text = " ".join([_clean_text(tt["text"]) for tt in toks])
                if not text: continue
                text = _strip_badges(text)
                bb = union_bbox_chain(toks)

                m = re.search(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+)", text)
                if m:
                    email = m.group(0).lower()
                    name = email_to_name.get(email)
                    if not name and attendee_names:
                        local = email.split("@")[0]
                        best = rf_process.extractOne(local, attendee_names, scorer=fuzz.WRatio)
                        if best and best[1] >= CONFIG["FUZZ_MIN_SCORE"]:
                            name = best[0]
                    if name:
                        tile_bb, _ = _tile_bbox_from_point_grid(bb[0] + bb[2]/2.0, bb[1] + bb[3]/2.0)
                        score = 85.0
                        if score > kept_score.get(name, -1):
                            kept_tiles[name] = tile_bb
                            kept_score[name] = score
                        continue

                if attendee_names:
                    cand = _normalize_label_for_match(text)
                    best = rf_process.extractOne(cand, attendee_names, scorer=fuzz.WRatio)
                    if best and best[1] >= CONFIG["FUZZ_MIN_SCORE"]:
                        name = best[0]
                        score = best[1]
                        tile_bb, _ = _tile_bbox_from_point_grid(bb[0] + bb[2]/2.0, bb[1] + bb[3]/2.0)
                        if score > kept_score.get(name, -1):
                            kept_tiles[name] = tile_bb
                            kept_score[name] = score

    # Collapse to one candidate per cell
    def _rc_from_bb(bb):
        cx = bb[0] + bb[2]/2.0; cy = bb[1] + bb[3]/2.0
        col = 0 if cx < col_bounds[1] else (1 if cx < col_bounds[2] else 2)
        row = 0 if cy < row_bounds[1] else (1 if cy < row_bounds[2] else 2)
        return int(row), int(col)

    by_cell = {}
    for name, bb in kept_tiles.items():
        rc = _rc_from_bb(bb)
        score = float(kept_score.get(name, 0.0))
        if rc not in by_cell or score > by_cell[rc][1]:
            by_cell[rc] = ((name, bb), score)

    kept = [by_cell[k][0] for k in sorted(by_cell.keys())]

    if CONFIG["DEBUG"]:
        overlay = img.copy()
        for x in map(int, col_bounds):
            cv2.line(overlay, (x, int(row_bounds[0])), (x, int(row_bounds[-1])), (255, 0, 255), 2)
        for y in map(int, row_bounds):
            cv2.line(overlay, (int(col_bounds[0]), y), (int(col_bounds[-1]), y), (255, 0, 255), 2)
        for name, bb in kept:
            x, y, w, h = bb
            cv2.rectangle(overlay, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(overlay, name[:20], (x+8, y+24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)
        cv2.imwrite(str(outdir / "tiles_from_tokens_v3_1.png"), overlay)

    return kept, (W, H), top_off

# ----------------------------- Blue mask & scoring -----------------------------
def _blue_mask_bgr(img_bgr: "np.ndarray") -> "np.ndarray":
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lo = np.array(CONFIG["LOWER_HSV"], dtype=np.uint8)
    hi = np.array(CONFIG["UPPER_HSV"], dtype=np.uint8)

    relax = CONFIG.get("RELAXED_BLUE_THRESHOLD", None)
    if relax is not None:
        lo_h, lo_s, lo_v = int(lo[0]), int(lo[1]), int(lo[2])
        hi_h, hi_s, hi_v = int(hi[0]), int(hi[1]), int(hi[2])
        center = (lo_h + hi_h) / 2.0
        width  = (hi_h - lo_h) * (1.0 + float(relax))
        new_lo = max(0,   int(center - width / 2))
        new_hi = min(179, int(center + width / 2))
        lo = np.array([new_lo, lo_s, lo_v], dtype=np.uint8)
        hi = np.array([new_hi, hi_s, hi_v], dtype=np.uint8)

    mask = cv2.inRange(hsv, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    return mask

# ----------------------------- Dynamic layout estimator -----------------------------
class DynLayoutEstimator:
    def __init__(self, period: int, max_side: int):
        self.period = max(1, int(period))
        self.max_side = max(2, int(max_side))
        self.last_update: int = -1
        self.col_bounds: Optional[List[int]] = None
        self.row_bounds: Optional[List[int]] = None

    def _best_centers(self, vals: List[float], kmax: int, lo: int, hi: int) -> Tuple[List[float], List[int]]:
        """Try k=2..kmax, choose centers with largest min-gap; return (centers, bounds)."""
        if len(vals) < 2:
            return [ (lo+hi)/2 ], [lo, hi]
        best_score, best_cs, best_bounds = -1.0, None, None
        for k in range(2, min(kmax, max(2, len(vals))) + 1):
            cs = _kmeans1d(vals, k=k)
            if not cs: continue
            cs = sorted(cs)
            gaps = [cs[i+1] - cs[i] for i in range(len(cs)-1)]
            score = min(gaps) if gaps else 0.0
            bounds = _bounds_from_centers_generic(cs, lo, hi)
            if score > best_score:
                best_score, best_cs, best_bounds = score, cs, bounds
        if best_bounds is None:
            best_bounds = [lo, hi]
        return best_cs or [ (lo+hi)/2 ], best_bounds

    def maybe_update(self, frame_bgr: np.ndarray, frame_idx: int):
        if self.last_update >= 0 and (frame_idx - self.last_update) < self.period:
            return
        self.last_update = frame_idx

        H, W = frame_bgr.shape[:2]
        y_lo_frac, y_hi_frac = CONFIG.get("LAYOUT_USE_BAND", (0.60, 0.92))
        y1 = int(H * y_lo_frac); y2 = max(y1 + 10, int(H * y_hi_frac))
        roi = _preproc_gray(frame_bgr[y1:y2, :])
        toks = _tess_data(roi, psm=6)
        toks = [t for t in toks if t["conf"] >= CONFIG["OCR_WORD_CONF_MIN"] and _clean_text(t["text"])]
        if len(toks) < 3:
            return  # not enough info; keep previous bounds

        xs = [(t["bbox"][0] + t["bbox"][2]/2.0) for t in toks]
        ys = [(y1 + t["bbox"][1] + t["bbox"][3]/2.0) for t in toks]

        _, col_bounds = self._best_centers(xs, self.max_side, 0, W)
        _, row_bounds = self._best_centers(ys, self.max_side, 0, H)

        # Keep exactly 4 bounds (3 columns/rows) if possible; else clamp to 3×3 compatible
        if len(col_bounds) >= 4:
            self.col_bounds = [col_bounds[0], col_bounds[1], col_bounds[2], col_bounds[-1]]
        else:
            self.col_bounds = [0, W//3, 2*W//3, W]
        if len(row_bounds) >= 4:
            self.row_bounds = [row_bounds[0], row_bounds[1], row_bounds[2], row_bounds[-1]]
        else:
            self.row_bounds = [0, H//3, 2*H//3, H]

    def slice(self, frame_bgr: np.ndarray) -> Dict[int, Tuple[np.ndarray, Tuple[int,int,int,int], Tuple[int,int]]]:
        H, W = frame_bgr.shape[:2]
        xs = self.col_bounds or [0, W//3, 2*W//3, W]
        ys = self.row_bounds or [0, H//3, 2*H//3, H]
        tiles: Dict[int, Tuple[np.ndarray, Tuple[int,int,int,int], Tuple[int,int]]] = {}
        tid = 0
        for r in range(3):
            for c in range(3):
                x1,x2 = int(xs[c]), int(xs[c+1])
                y1,y2 = int(ys[r]), int(ys[r+1])
                tiles[tid] = (frame_bgr[y1:y2, x1:x2], (x1, y1, x2-x1, y2-y1), (r, c))
                tid += 1
        return tiles

# ----------------------------- Scoring helpers -----------------------------
def _score_tiles_by_mask_with_bounds(frame_bgr: np.ndarray, tiles: Dict[int, Tuple[np.ndarray, Tuple[int,int,int,int], Tuple[int,int]]]) -> Tuple[Optional[int], float, np.ndarray]:
    mask = _blue_mask_bgr(frame_bgr)
    H, W = frame_bgr.shape[:2]
    total_area = H * W
    min_area = max(1.0, float(CONFIG.get("MIN_EDGE_AREA_FRAC", 0.0005)) * total_area)
    best_tid, best_score = None, 0.0
    for tid, (_, (x,y,w,h), _) in tiles.items():
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(W, x + w); y1 = min(H, y + h)
        if x1 <= x0 or y1 <= y0: continue
        tile_mask = mask[y0:y1, x0:x1]
        score = float((tile_mask > 0).sum())
        if score >= min_area and score > best_score:
            best_score, best_tid = score, tid
    return best_tid, best_score, mask

# ----------------------------- Alignment & Normalization -----------------------------
VOCAB_RULES = [
    (r"\bF[-\s]?out\b", "ephOut"),
    (r"\bFL\b", "ephOut"),
    (r"\bsat[-\s]?no\b", "SATNO"),
    (r"\bOmitron\b", "Omitron"),
    (r"\b[Ee]phemerides\b", "ephemeris"),
]

def apply_vocab_rules(text: str) -> str:
    out = text
    for pat, repl in VOCAB_RULES: out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    return out

def map_tile_to_attendee_name(tile_name: str, attendees: Dict[str, str]) -> str:
    if not attendees:
        return tile_name
    choices = list(attendees.keys())
    if not choices:
        return tile_name
    generic = len(tile_name.split()) <= 1 or len(tile_name) <= 6
    thresh = 80 if generic else 70
    best = rf_process.extractOne(tile_name, choices, scorer=fuzz.WRatio)
    if best and best[1] >= thresh:
        return best[0]
    return tile_name

# ----------------------------- Segment attribution (vote) -----------------------------
def overlap(a, b):
    s1, e1 = a; s2, e2 = b
    s = max(s1, s2); e = min(e1, e2)
    return max(0.0, e - s)

def attribute_segments(segments: List[Dict], timeline: List[SpeakerEvent], attendees: Dict[str, str]) -> List['TranscriptSegment']:
    if not segments: return []
    events = sorted(timeline, key=lambda e: e.start)
    out: List[TranscriptSegment] = []
    i = 0
    for seg in segments:
        s_start, s_end, s_text = float(seg["start"]), float(seg["end"]), apply_vocab_rules(seg["text"])
        s_prob = seg.get("prob", None)
        pad = 0.25
        while i < len(events) and events[i].end < s_start - pad: i += 1
        j = i; counts: Dict[str, float] = {}
        while j < len(events) and events[j].start <= s_end + pad:
            ev = events[j]
            dur = overlap((s_start - pad, s_end + pad), (ev.start, ev.end))
            if dur > 0:
                counts[ev.tile_name] = counts.get(ev.tile_name, 0.0) + dur
            j += 1
        if not counts:
            out.append(TranscriptSegment(start=s_start, end=s_end, text=s_text, speaker=None, validated=False, prob=s_prob)); continue
        chosen_name = max(counts.items(), key=lambda kv: kv[1])[0]
        mapped = map_tile_to_attendee_name(chosen_name, attendees)
        out.append(TranscriptSegment(start=s_start, end=s_end, text=s_text, speaker=mapped, validated=True, prob=s_prob))
    return out

def attribute_segments_transcript_only(segments: List[Dict], placeholder: str) -> List['TranscriptSegment']:
    out: List[TranscriptSegment] = []
    for seg in segments:
        s_start, s_end, s_text = float(seg["start"]), float(seg["end"]), apply_vocab_rules(seg["text"])
        s_prob = seg.get("prob", None)
        out.append(TranscriptSegment(start=s_start, end=s_end, text=s_text, speaker=placeholder, validated=False, prob=s_prob))
    return out

# ----------------------------- Perspective warp helper -----------------------------
def _maybe_warp(frame: np.ndarray) -> np.ndarray:
    quad = CONFIG.get("WARP_QUAD")
    size = CONFIG.get("WARP_SIZE")
    if not quad or not size:
        return frame
    dstW, dstH = int(size[0]), int(size[1])
    if dstW <= 0 or dstH <= 0:
        return frame
    try:
        src = np.float32(quad)
        dst = np.float32([(0,0),(dstW,0),(dstW,dstH),(0,dstH)])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame, M, (dstW, dstH))
    except Exception as e:
        print(f"[WARP] failed: {e}")
        return frame

# ----------------------------- Dynamic grid orchestrator -----------------------------
class DynamicGridAttributor:
    def __init__(self, rows_max: int = 4, verbose: bool = True):
        self.rows_max = max(1, rows_max)
        self.verbose = verbose
        self.speaker_map: Dict[int, Dict[str, Any]] = {}
        self.frame_count = 0

    def log(self, msg: str):
        if self.verbose:
            print(f"[DynGrid] {msg}")

    def detect_grid_size(self, frame: np.ndarray) -> Tuple[int, int]:
        # We keep 3×3 slicing; bounds come from DynLayoutEstimator
        return 3, 3

    def name_from_tile(self, tile_img: np.ndarray, attendees: Dict[str,str], initials_map: Dict[str, List[str]]) -> Optional[str]:
        if not CONFIG["ONFRAME_OCR"]:
            return None
        h, w = tile_img.shape[:2]
        frac = float(CONFIG.get("BOTTOM_STRIP_FRAC", 0.55))
        y1 = int(h * frac)
        y2 = max(y1 + 5, h - 2)
        roi = _preproc_gray(tile_img[y1:y2, :])
        toks = _tess_data(roi, psm=7)
        texts = [t for t in toks if t["conf"] >= CONFIG["OCR_WORD_CONF_MIN"] and t["text"]]
        if not texts:
            return None
        texts.sort(key=lambda z: (z["bbox"][1], z["bbox"][0]))
        line = " ".join([_clean_text(t["text"]) for t in texts])
        line = _strip_badges(line)
        if not line:
            return None
        if attendees:
            cand = _normalize_label_for_match(line)
            best = rf_process.extractOne(cand, list(attendees.keys()), scorer=fuzz.WRatio)
            if best and best[1] >= CONFIG["FUZZ_MIN_SCORE"]:
                return best[0]
        for t in texts:
            token = _clean_text(t["text"]).upper()
            if re.fullmatch(r"[A-Z]{2}", token):
                names = initials_map.get(token, [])
                if len(names) == 1:
                    return names[0]
        return None

    def update_map(self, tid: int, name: Optional[str]):
        self.speaker_map.setdefault(tid, {"name": None, "last_seen": -1})
        if name:
            self.speaker_map[tid]["name"] = name
        self.speaker_map[tid]["last_seen"] = self.frame_count

    def prune_stale(self, ttl: int):
        doomed = [tid for tid, meta in self.speaker_map.items() if (self.frame_count - meta.get("last_seen", -1)) > ttl]
        for tid in doomed:
            self.speaker_map.pop(tid, None)

# ----------------------------- Orchestrator -----------------------------
def build_active_speaker_timeline(
    video_path: Path,
    tiles: List[Tile],
    fps: float,
    video_workers: int,
    debug_dir: Optional[Path],
    dynamic_mode: bool = False,
    attendees: Optional[Dict[str, str]] = None,
    initials_map: Optional[Dict[str, List[str]]] = None,
    dynamic_export_path: Optional[Path] = None
) -> Tuple[List[SpeakerEvent], Optional[DynamicGridAttributor]]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    crop = CONFIG["VIDEO_CROP"]

    def crop_frame(f):
        if not crop:
            return f
        x, y, w, h = crop
        return f[y:y + h, x:x + w]

    step = int(max(1, round(native_fps / max(0.1, fps))))
    idx = 0
    jobs = []

    dyn = DynamicGridAttributor(rows_max=CONFIG["DYNAMIC_MAX_SIDE"], verbose=bool(CONFIG["DEBUG"])) if dynamic_mode else None
    layout = DynLayoutEstimator(period=CONFIG["LAYOUT_UPDATE_PERIOD"], max_side=CONFIG["DYNAMIC_MAX_SIDE"]) if dynamic_mode else None

    def _detect_one_frame(ts: float, frame):
        frame = crop_frame(frame)
        frame = _maybe_warp(frame)
        if dynamic_mode and dyn is not None and layout is not None:
            rows, cols = dyn.detect_grid_size(frame)  # 3×3 logical grid
            best_tid = None; best_score = 0.0
            layout.maybe_update(frame, int(ts * 1000))  # use ms as monotonically increasing index
            tiles_dyn = layout.slice(frame)
            best_tid, best_score, mask_dbg = _score_tiles_by_mask_with_bounds(frame, tiles_dyn)
            return ts, ("dynamic", best_tid, best_score, tiles_dyn), frame if CONFIG["DEBUG"] else None, mask_dbg
        else:
            # static: use provided tiles (scaled earlier)
            mask_dbg = _blue_mask_bgr(frame)
            H, W = frame.shape[:2]
            total_area = H * W
            min_area = max(1.0, float(CONFIG.get("MIN_EDGE_AREA_FRAC", 0.0005)) * total_area)
            best_name, best_score = None, 0.0
            for t in tiles:
                x, y, w, h = t.bbox
                x0 = max(0, x); y0 = max(0, y)
                x1 = min(W, x + w); y1 = min(H, y + h)
                if x1 <= x0 or y1 <= y0: continue
                tile_mask = mask_dbg[y0:y1, x0:x1]
                score = float((tile_mask > 0).sum())
                if score >= min_area and score > best_score:
                    best_score = score
                    best_name = t.name
            return ts, ("static", best_name, best_score, None), frame if CONFIG["DEBUG"] else None, mask_dbg

    with ThreadPoolExecutor(max_workers=max(1, video_workers)) as ex:
        while True:
            ret = cap.grab()
            if not ret: break
            if idx % step == 0:
                ret, frame = cap.retrieve()
                if not ret or frame is None: break
                ts = idx / native_fps
                jobs.append(ex.submit(_detect_one_frame, ts, frame))
            idx += 1

        raw_events: List[SpeakerEvent] = []
        debug_counter = 0

        for fut in as_completed(jobs):
            ts, result, frame_dbg, mask_dbg = fut.result()
            if result[0] == "dynamic" and dyn is not None and layout is not None:
                _, best_tid, best_score, tiles_dyn = result
                dyn.frame_count += 1
                dyn.prune_stale(CONFIG["DYNAMIC_NAME_CACHE_TTL"])
                if best_tid is not None and best_tid in tiles_dyn:
                    tile_img, tile_bb, (rr, cc) = tiles_dyn[best_tid]
                    name = dyn.name_from_tile(tile_img, attendees or {}, initials_map or {}) or f"Tile{rr}{cc}"
                    dyn.update_map(best_tid, name)
                    raw_events.append(SpeakerEvent(start=ts, end=ts, tile_name=name, validated=True))
                if CONFIG["DEBUG"] and debug_dir and frame_dbg is not None and tiles_dyn:
                    if debug_counter % CONFIG["DEBUG_EVERY_N"] == 0:
                        canvas = frame_dbg.copy()
                        # draw grid bounds
                        for _, bb, (r,c) in tiles_dyn.values():
                            x,y,w,h = bb
                            color = (0,255,0)
                            cv2.rectangle(canvas, (x,y), (x+w,y+h), color, 2)
                        cv2.imwrite(str(debug_dir / f"frame_{int(ts*1000):08d}.jpg"), canvas)
                        if mask_dbg is not None:
                            cv2.imwrite(str(debug_dir / f"mask_{int(ts*1000):08d}.png"), mask_dbg)
                    debug_counter += 1
            else:
                _, best_name, best_score, _ = result
                if best_name is not None:
                    raw_events.append(SpeakerEvent(start=ts, end=ts, tile_name=best_name, validated=True))
                if CONFIG["DEBUG"] and debug_dir and frame_dbg is not None:
                    if debug_counter % CONFIG["DEBUG_EVERY_N"] == 0:
                        canvas = frame_dbg.copy()
                        for t in tiles:
                            x, y, w, h = t.bbox
                            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.imwrite(str(debug_dir / f"frame_{int(ts*1000):08d}.jpg"), canvas)
                        if mask_dbg is not None:
                            cv2.imwrite(str(debug_dir / f"mask_{int(ts*1000):08d}.png"), mask_dbg)
                    debug_counter += 1

    cap.release()

    if dynamic_mode and dyn is not None and CONFIG["EXPORT_DYNAMIC_MAP"] and dynamic_export_path:
        try:
            with open(dynamic_export_path, "w", encoding="utf-8") as f:
                json.dump(dyn.speaker_map, f, indent=2)
            print(f"[DynGrid] Exported dynamic map -> {dynamic_export_path}")
        except Exception as e:
            print(f"[DynGrid] Failed exporting dynamic map: {e}")

    if not raw_events:
        return [], dyn

    # Merge and smooth pings
    raw_events.sort(key=lambda e: e.start)
    merged: List[SpeakerEvent] = []
    cur = raw_events[0]
    merge_tol = (1.5 / max(0.1, fps))
    for ev in raw_events[1:]:
        if ev.tile_name == cur.tile_name and ev.start - cur.end <= merge_tol:
            cur.end = ev.end
        else:
            merged.append(cur); cur = ev
    merged.append(cur)

    smoothed: List[SpeakerEvent] = []
    prev: Optional[SpeakerEvent] = None
    gap_tol = (1.0 / max(0.1, fps))
    for ev in merged:
        if prev and ev.start - prev.end < gap_tol:
            prev.end = ev.end
        else:
            if prev: smoothed.append(prev)
            prev = dataclasses.replace(ev)
    if prev:
        smoothed.append(prev)

    return smoothed, dyn

# ----------------------------- Outputs -----------------------------
def write_vtt(segments, out_path: Path, attendees: Dict[str, str]):
    out_lines = ["WEBVTT",""]
    for i, seg in enumerate(segments, 1):
        speaker = seg.speaker or "UNKNOWN"
        company = attendees.get(seg.speaker or "", "")
        star = "*" if seg.validated and seg.speaker else ""
        speaker_line = f"{speaker}{star} ({company})" if company else f"{speaker}{star}"
        out_lines += [str(i), f"{hhmmss(seg.start)} --> {hhmmss(seg.end)}", f"{speaker_line}: {seg.text.strip()}", ""]
    out_path.write_text("\n".join(out_lines), encoding="utf-8")

def write_segments_json(segments, out_path: Path):
    data = [dataclasses.asdict(s) for s in segments]
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def write_qa_report(segments, timeline, out_path: Path):
    total = len(segments)
    attributed = sum(1 for s in segments if s.speaker is not None)
    validated = sum(1 for s in segments if s.validated and s.speaker is not None)
    coverage = {
        "segments_total": total,
        "segments_with_speaker": attributed,
        "segments_with_visual_validation": validated,
        "attribution_rate": round(attributed / total, 4) if total else 0.0,
        "visual_validation_rate": round(validated / max(1, attributed), 4) if attributed else 0.0,
        "speaker_windows_found": len(timeline),
    }
    out_path.write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    print("[REPORT]", json.dumps(coverage))

# ----------------------------- Interactive -----------------------------
def _try_tk():
    try:
        import tkinter as _tk
        from tkinter import filedialog as _fd
        return _tk, _fd
    except Exception:
        return None, None

def _prompt_path_console(label: str, is_dir: bool = False) -> Path:
    print(f"[INPUT] {label} (drag the file/folder here then press Enter)")
    while True:
        raw = input("> ").strip().strip('"').strip("'")
        if not raw:
            print("Please provide a path."); continue
        pth = Path(raw).expanduser().resolve()
        if is_dir and pth.is_dir(): return pth
        if not is_dir and pth.is_file(): return pth
        print("Not valid, try again.")

def _prompt_path_gui(label: str, is_dir: bool = False, filetypes=None) -> Optional[Path]:
    _tk, _fd = _try_tk()
    if not _tk: return None
    try:
        root = _tk.Tk(); root.withdraw()
        path_str = ""
        if is_dir: path_str = _fd.askdirectory(title=label) or ""
        else: path_str = _fd.askopenfilename(title=label, filetypes=filetypes or [("All files", "*.*")]) or ""
        root.update_idletasks(); root.destroy()
        if path_str: return Path(path_str).expanduser().resolve()
    except Exception:
        return None
    return None

def prompt_paths_interactive(args) -> Tuple[Path, Optional[Path], Path, Path]:
    video = args.video if isinstance(args.video, Path) else None
    screenshot = args.screenshot if isinstance(args.screenshot, Path) else None
    ics = args.ics if isinstance(args.ics, Path) else None
    outdir = args.outdir if isinstance(args.outdir, Path) else None

    if not video:
        video = _prompt_path_gui("Select meeting video (mp4/mkv/etc.)",
                                 filetypes=[("Video", "*.mp4 *.mkv *.mov *.avi"), ("All", "*.*")]) \
                or _prompt_path_console("Drop meeting video path")
    if not screenshot:
        screenshot = _prompt_path_gui("Select meeting grid screenshot (PNG/JPG) [optional]",
                                      filetypes=[("Images", "*.png *.jpg *.jpeg"), ("All", "*.*")])
        if not screenshot:
            print("[INPUT] No screenshot selected (ok). Press Enter to continue without, or paste a path to use one.")
            raw = input("> ").strip().strip('"').strip("'")
            if raw:
                p = Path(raw).expanduser().resolve()
                if p.is_file():
                    screenshot = p
    if not ics:
        ics = _prompt_path_gui("Select meeting .ics file", filetypes=[("ICS", "*.ics"), ("All", "*.*")]) \
              or _prompt_path_console("Drop .ics path")
    if not outdir:
        outdir = _prompt_path_gui("Select output directory", is_dir=True) \
                 or _prompt_path_console("Drop output directory", is_dir=True)

    return video, screenshot, ics, outdir

# ----------------------------- Main -----------------------------
def main():
    p = argparse.ArgumentParser(description="V3: mask-per-tile scoring + hue relax + segment voting")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--video", type=Path)
    p.add_argument("--screenshot", type=Path)  # optional
    p.add_argument("--ics", type=Path)
    p.add_argument("--outdir", type=Path)
    p.add_argument("--dynamic-grid", action="store_true", help="Enable dynamic grid mode")
    p.add_argument("--relaxed-blue-threshold", type=float, default=None, help="Widen HSV hue band by this factor (e.g., 0.35)")
    args = p.parse_args()

    # Toggle dynamic grid via ENV, and set relax factor
    env_val = os.getenv("ATTRIB_DYNAMIC_GRID")
    if env_val is not None:
        CONFIG["DYNAMIC_GRID"] = env_val.strip().lower() in ("1", "true", "yes", "on")

    if args.relaxed_blue_threshold is not None:
        CONFIG["RELAXED_BLUE_THRESHOLD"] = float(args.relaxed_blue_threshold)

    if args.interactive or not args.video or not args.ics or not args.outdir:
        args.video, args.screenshot, args.ics, args.outdir = prompt_paths_interactive(args)

    if args.dynamic_grid:
        CONFIG["DYNAMIC_GRID"] = True

    try:
        import torch
        print(f"[CUDA] torch.cuda.is_available() = {torch.cuda.is_available()}")
    except Exception:
        pass

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Parse ICS (extended)
    name_to_company, email_to_name, initials_map = parse_ics_attendees_extended(args.ics)
    print(f"[ICS] Attendees parsed: {len(name_to_company)} names, {len(email_to_name)} emails")

    tiles = []
    shot_size = None

    # OCR → tiles (if screenshot provided)
    if args.screenshot and Path(args.screenshot).exists():
        try:
            kept, shot_size, top_off = ocr_to_tiles(args.screenshot, name_to_company, email_to_name, initials_map, outdir=outdir)
            print(f"[OCR] Tiles from tokens: {len(kept)} | screenshot size={shot_size}")
            if kept[:9]:
                print("     Sample:", [n for n,_ in kept[:9]])
            tiles = [Tile(name=n, bbox=bb, grid_pos=(0,0)) for n,bb in kept]
            tiles.sort(key=lambda t: (t.bbox[1], t.bbox[0]))
            for i, t in enumerate(tiles):
                tiles[i] = Tile(name=t.name, bbox=t.bbox, grid_pos=(i//3, i%3))
            print(f"[GRID] Tiles built (static): {len(tiles)}")
        except Exception as e:
            print(f"[OCR] Failed to build tiles from screenshot: {e}")
            tiles = []
    else:
        print("[OCR] No screenshot provided.")

    # Video size
    cap_sz = cv2.VideoCapture(str(args.video))
    if not cap_sz.isOpened(): raise RuntimeError("Cannot open video to query size")
    vw = int(cap_sz.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    vh = int(cap_sz.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap_sz.release()
    dst_size = (vw, vh) if not CONFIG["VIDEO_CROP"] else (CONFIG["VIDEO_CROP"][2], CONFIG["VIDEO_CROP"][3])
    print(f"[VIDEO] size={vw}x{vh} | target grid size={dst_size} | crop={CONFIG['VIDEO_CROP']}")

    tiles_scaled = []
    if tiles and shot_size:
        tiles_scaled = scale_tiles_to_frame(tiles, shot_size, dst_size)

    # Audio → STT
    wav_path = outdir / "audio_16k.wav"
    run_ffmpeg_extract_audio(args.video, wav_path)
    stt_segments, used = safe_transcribe(wav_path, language=CONFIG["LANGUAGE"])
    print(f"[STT] Using: {used} | Segments: {len(stt_segments)}")

    # Timeline & debug
    debug_dir = outdir / "debug" if CONFIG["DEBUG"] else None
    if debug_dir: debug_dir.mkdir(parents=True, exist_ok=True)

    dynamic_export_path = outdir / "dynamic_speaker_map.json" if CONFIG["EXPORT_DYNAMIC_MAP"] else None

    use_dynamic = CONFIG["DYNAMIC_GRID"] or (CONFIG["FALLBACK_IF_NO_SCREENSHOT"] and not tiles_scaled)
    timeline, dyn = build_active_speaker_timeline(
        video_path=args.video,
        tiles=tiles_scaled,
        fps=CONFIG["FPS_SAMPLE"],
        video_workers=CONFIG["VIDEO_WORKERS"],
        debug_dir=debug_dir,
        dynamic_mode=use_dynamic,
        attendees=name_to_company,
        initials_map=initials_map,
        dynamic_export_path=dynamic_export_path
    )
    print(f"[TL] Active-speaker events: {len(timeline)} (mode={'dynamic' if use_dynamic else 'static'})")

    # Attribute & outputs
    if use_dynamic and not timeline and CONFIG["FALLBACK_IF_NO_SCREENSHOT"]:
        print("[ATTRIB] No events; using transcript-only mode.")
        segments = attribute_segments_transcript_only(stt_segments, CONFIG["TRANSCRIPT_ONLY_PLACEHOLDER"])
    else:
        segments = attribute_segments(stt_segments, timeline, name_to_company)

    attributed = sum(1 for s in segments if s.speaker)
    validated = sum(1 for s in segments if s.speaker and s.validated)
    print(f"[ATTRIB] segments={len(segments)} | with_speaker={attributed} | visual_validated={validated}")

    vtt_path = outdir / "speaker_attributed.vtt"
    json_path = outdir / "transcript_segments.json"
    qa_path = outdir / "qa_report.json"
    write_vtt(segments, vtt_path, name_to_company)
    write_segments_json(segments, json_path)
    write_qa_report(segments, timeline, qa_path)

    if len(timeline) == 0:
        print("[DIAG] No blue-border events. Try adjusting HSV or --relaxed-blue-threshold, or set VIDEO_CROP.")

    print(json.dumps({
        "status": "ok",
        "mode": "dynamic" if use_dynamic else "static",
        "outputs": {
            "vtt": str(vtt_path),
            "segments_json": str(json_path),
            "qa_report": str(qa_path),
            "dynamic_map": str(dynamic_export_path) if dynamic_export_path else None
        }
    }, indent=2))

if __name__ == "__main__":
    main()
