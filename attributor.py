#!/usr/bin/env python3
"""
attributor.py  (V2_8_1)

Purpose
-------
Offline Teams/Zoom-style meeting attribution:
- Parse .ics attendees (names, emails, companies)
- OCR screenshot (tokens + cell-bottom labels) → infer 3×3 tiles
- Detect active speaker via blue border and attribute transcript segments
- Generate VTT + JSON + QA report
- GPU-first STT (faster-whisper), with CPU fallbacks

This patch (V2_8_1) applies:
- FUZZ_MIN_SCORE: 60 → **55**
- INITIALS_CONF_MIN: 60 → **50**
- **Large-bubble initials** heuristic (accept big 2-letter tokens inside cells)
- Keeps auto top-offset, email→name normalization, phrase assembly (1..5 tokens)

Run
---
    python attributor.py --interactive
"""

from __future__ import annotations
import argparse, dataclasses, json, os, re, subprocess, sys, math
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ============================== CONFIG ==============================
CONFIG = {
    "LANGUAGE": "en",
    "PREFER_FASTER": True,
    "STT_MODEL": "medium",
    "DEVICE": "cuda",
    "COMPUTE_TYPE": "float16",
    "CPU_THREADS": 0,
    "WHISPER_WORKERS": 2,

    "FPS_SAMPLE": 3.0,
    "VIDEO_WORKERS": 4,

    # Blue border HSV (Teams blue)
    "LOWER_HSV": (95, 70, 70),
    "UPPER_HSV": (135, 255, 255),

    "MIN_BOX_W": 30,
    "MIN_BOX_H": 30,
    "MIN_AREA": 150,
    "MAX_STROKE_RATIO": 0.30,

    "VIDEO_CROP": None,           # (x,y,w,h) or None

    "TESSERACT_CMD": None,

    "DEBUG": True,
    "DEBUG_EVERY_N": 15,

    # OCR tuning
    "OCR_LANG": "eng",
    "OCR_ALLOWED_RE": r"[^A-Za-z0-9 .,'()@\-\[\]]",
    "OCR_MIN_LEN": 2,
    "FUZZ_MIN_SCORE": 55,         # relaxed
    "OCR_WORD_CONF_MIN": 50,
    "INITIALS_CONF_MIN": 50,      # relaxed
    "WORDS_Y_MIN_FRAC": 0.35,

    # Grid geometry
    "TOP_OFFSET_FRAC": 0.09,
    "AUTO_TOP_OFFSET": True,
    "GRID_INSET": 6,

    # Large-bubble initials heuristic
    "LARGE_BUBBLE_AREA_FRAC": 0.035,  # ~3.5% of cell area
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
    txt = re.sub(CONFIG["OCR_ALLOWED_RE"], "", txt)  # strip disallowed chars
    txt = re.sub(r"\s{2,}", " ", txt)
    return txt.strip()

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
    if not CONFIG["AUTO_TOP_OFFSET"]: return int(CONFIG["TOP_OFFSET_FRAC"] * gray.shape[0])
    H, W = gray.shape[:2]
    # gradient magnitude per row
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    row_energy = mag.mean(axis=1)
    search_to = int(0.2 * H)
    if search_to < 10: return int(CONFIG["TOP_OFFSET_FRAC"] * H)
    idx = np.argmax(row_energy[:search_to])
    off = int(min(max(idx + 10, H*0.06), H*0.15))
    return off

def _cell_bbox_from_point(cx, cy, W, H, top_off):
    grid_h = max(1, H - top_off)
    tile_w = W / 3.0
    tile_h = grid_h / 3.0
    col = int(max(0, min(2, cx // tile_w)))
    row = int(max(0, min(2, (cy - top_off) // tile_h)))
    bx = int(col * tile_w); by = int(top_off + row * tile_h)
    bw = int(tile_w); bh = int(tile_h)
    inset = CONFIG["GRID_INSET"]
    return (bx+inset, by+inset, max(1,bw-2*inset), max(1,bh-2*inset)), (row, col)

def _tile_bbox_from_bbox(bb, W, H, top_off):
    x,y,w,h = bb
    cx = x + w/2; cy = y + h/2
    return _cell_bbox_from_point(cx, cy, W, H, top_off)

def _union_bbox(b1, b2):
    x1,y1,w1,h1 = b1; x2,y2,w2,h2 = b2
    xa = min(x1,x2); ya = min(y1,y2)
    xb = max(x1+w1, x2+w2); yb = max(y1+h1, y2+h2)
    return (xa, ya, xb-xa, yb-ya)

def _strip_badges(t: str) -> str:
    t = re.sub(r"\[[^\]]+\]", " ", t)
    t = re.sub(r"\([^\)]+\)", " ", t)
    return re.sub(r"\s{2,}", " ", t).strip()

def ocr_to_tiles(screenshot_path: Path, attendees_name_to_company: Dict[str,str],
                 email_to_name: Dict[str,str], initials_map: Dict[str,List[str]], outdir: Path):
    set_tesseract_cmd_if_provided()
    img = cv2.imread(str(screenshot_path))
    if img is None: raise RuntimeError(f"Failed to read screenshot: {screenshot_path}")
    H, W = img.shape[:2]
    gray = _preproc_gray(img)
    top_off = _auto_top_offset(gray)
    print(f"[GRID] top_offset={top_off} (auto={'on' if CONFIG['AUTO_TOP_OFFSET'] else 'off'})")

    # Pass A: full image PSM6
    wordsA = _tess_data(gray, psm=6)
    # Pass B: bottom strips per cell PSM7
    strips = []
    grid_h = H - top_off
    tile_h = int(grid_h / 3.0)
    tile_w = int(W / 3.0)
    for r in range(3):
        y1 = top_off + r*tile_h + int(tile_h*0.72)
        y2 = top_off + (r+1)*tile_h - int(tile_h*0.05)
        y1 = max(top_off, min(H-1, y1)); y2 = max(y1+5, min(H, y2))
        strips.append((y1, y2))
    wordsB = []
    for r in range(3):
        for c in range(3):
            x1 = c*tile_w; x2 = (c+1)*tile_w
            y1,y2 = strips[r]
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0: continue
            data = _tess_data(roi, psm=7)
            for w in data:
                x,y,wid,hei = w["bbox"]
                w["bbox"] = (x1+x, y1+y, wid, hei)
            wordsB.extend(data)

    # Merge + filter
    words_all = wordsA + wordsB
    words_keep = [w for w in words_all if w["conf"] >= CONFIG["OCR_WORD_CONF_MIN"] and _clean_text(w["text"])]
    # Debug overlay
    dbg = img.copy()
    for w in words_keep:
        x,y,ww,hh = w["bbox"]
        cv2.rectangle(dbg, (x,y), (x+ww,y+hh), (0,255,255), 1)
    cv2.imwrite(str(outdir / "ocr_tokens_v2.png"), dbg)
    print(f"[OCRDBG] Tokens (merged) conf>={CONFIG['OCR_WORD_CONF_MIN']}: {len(words_keep)}")

    name_to_company = attendees_name_to_company
    attendee_names = list(name_to_company.keys())

    kept_tiles: Dict[str, Tuple[int,int,int,int]] = {}
    kept_score: Dict[str, float] = {}

    # Initials-first (global) + large-bubble heuristic
    init_hits = 0; init_seen = 0
    for w in words_keep:
        t = _clean_text(w["text"]).upper()
        if re.fullmatch(r"[A-Z]{2}", t):
            init_seen += 1
            # cell bbox to estimate area
            tile_bb, _ = _tile_bbox_from_bbox(w["bbox"], W, H, top_off)
            cell_area = tile_bb[2] * tile_bb[3]
            token_area = w["bbox"][2] * w["bbox"][3]
            large_bubble = token_area >= CONFIG["LARGE_BUBBLE_AREA_FRAC"] * cell_area
            conf_ok = (w["conf"] >= CONFIG["INITIALS_CONF_MIN"])
            if conf_ok or large_bubble:
                names = initials_map.get(t, [])
                if len(names) == 1:
                    name = names[0]
                    score = max(w["conf"], 80.0 if large_bubble else w["conf"])
                    if score > kept_score.get(name, -1):
                        kept_tiles[name] = tile_bb; kept_score[name] = score
                        init_hits += 1
    print(f"[OCRDBG] Initials tokens seen={init_seen} mapped_unique={init_hits}")

    # Phrase assembly per cell bottom (1..5)
    def token_in_bottom_strip(w):
        x,y,ww,hh = w["bbox"]
        cy = y + hh/2
        return cy >= top_off + tile_h*0.70

    cell_words = [w for w in words_keep if token_in_bottom_strip(w)]
    cell_words.sort(key=lambda z: (z["bbox"][1], z["bbox"][0]))

    lines: List[List[Dict[str,Any]]] = []
    for w in cell_words:
        if not lines: lines.append([w]); continue
        last_y = sum([ww["bbox"][1] for ww in lines[-1]])/len(lines[-1])
        if abs(w["bbox"][1]-last_y) <= 22:
            lines[-1].append(w)
        else:
            lines.append([w])

    def union_bbox_chain(toks):
        bb = toks[0]["bbox"]
        for t in toks[1:]:
            bb = _union_bbox(bb, t["bbox"])
        return bb

    phrase_attempts = 0; phrase_hits = 0
    for line in lines:
        line.sort(key=lambda z: z["bbox"][0])
        n = len(line)
        for i in range(n):
            for L in (1,2,3,4,5):
                if i+L>n: break
                toks = line[i:i+L]
                phrase_attempts += 1
                text = " ".join([_clean_text(tt["text"]) for tt in toks])
                if not text: continue
                text = _strip_badges(text)
                bb = union_bbox_chain(toks)

                # Email → name
                email_match = re.search(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+)", text)
                if email_match:
                    email = email_match.group(0).lower()
                    name = email_to_name.get(email)
                    if not name and attendee_names:
                        local = email.split("@")[0]
                        best = rf_process.extractOne(local, attendee_names, scorer=fuzz.WRatio)
                        if best and best[1] >= CONFIG["FUZZ_MIN_SCORE"]:
                            name = best[0]
                    if name:
                        tile_bb, _ = _tile_bbox_from_bbox(bb, W, H, top_off)
                        score = 85.0
                        if score > kept_score.get(name, -1):
                            kept_tiles[name] = tile_bb; kept_score[name] = score
                            phrase_hits += 1
                        continue

                # Fuzzy full-name
                if attendee_names:
                    best = rf_process.extractOne(text, attendee_names, scorer=fuzz.WRatio)
                    if best and best[1] >= CONFIG["FUZZ_MIN_SCORE"]:
                        name = best[0]; score = best[1]
                        tile_bb, _ = _tile_bbox_from_bbox(bb, W, H, top_off)
                        if score > kept_score.get(name, -1):
                            kept_tiles[name] = tile_bb; kept_score[name] = score
                            phrase_hits += 1

    print(f"[OCRDBG] Phrase attempts={phrase_attempts} accepted={phrase_hits}")

    kept = [(name, kept_tiles[name]) for name in kept_tiles.keys()]
    # overlay
    overlay = img.copy()
    for name, bb in kept:
        x,y,w,h = bb
        cv2.rectangle(overlay, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(overlay, name[:20], (x+8,y+24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)
    out_tiles = outdir / "tiles_from_tokens_v2.png"
    cv2.imwrite(str(out_tiles), overlay)
    print(f"[OCRDBG] Kept attendees as tiles: {len(kept)} -> {out_tiles}")

    return kept, (W,H), top_off

# ----------------------------- Blue Border Detection -----------------------------
def detect_blue_border_regions(frame: "np.ndarray") -> Tuple[List[Tuple[int,int,int,int]], "np.ndarray"]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array(CONFIG["LOWER_HSV"], dtype=np.uint8)
    upper = np.array(CONFIG["UPPER_HSV"], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if w < CONFIG["MIN_BOX_W"] or h < CONFIG["MIN_BOX_H"]: continue
        if area < CONFIG["MIN_AREA"]: continue
        stroke_ratio = area / max(1.0, w * h)
        if stroke_ratio > CONFIG["MAX_STROKE_RATIO"]: continue
        boxes.append((x, y, w, h))
    return boxes, mask

def match_boxes_to_tiles(boxes, tiles: List[Tile]) -> Optional[Tile]:
    if not boxes or not tiles: return None
    def iou(a, b):
        ax, ay, aw, ah = a; bx, by, bw, bh = b
        a2 = (ax+aw, ay+ah); b2 = (bx+bw, by+bh)
        x_left = max(ax, bx); y_top = max(ay, by)
        x_right = min(a2[0], b2[0]); y_bottom = min(a2[1], b2[1])
        if x_right <= x_left or y_bottom <= y_top: return 0.0
        inter = (x_right-x_left)*(y_bottom-y_top)
        union = aw*ah + bw*bh - inter
        return inter / max(1e-6, union)
    best, best_iou = None, 0.0
    for b in boxes:
        for t in tiles:
            i = iou(b, t.bbox)
            if i > best_iou:
                best_iou, best = i, t
    return best if best_iou >= 0.05 else None

def scale_tiles_to_frame(tiles: List[Tile], src_size: Tuple[int,int], dst_size: Tuple[int,int]) -> List[Tile]:
    sw, sh = src_size; dw, dh = dst_size
    if sw == 0 or sh == 0: return tiles
    sx, sy = dw / sw, dh / sh
    out = []
    for t in tiles:
        x, y, w, h = t.bbox
        out.append(Tile(
            name=t.name,
            bbox=(int(round(x*sx)), int(round(y*sy)), int(round(w*sx)), int(round(h*sy))),
            grid_pos=t.grid_pos
        ))
    return out

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
    if not attendees: return tile_name
    choices = list(attendees.keys())
    if not choices: return tile_name
    best = rf_process.extractOne(tile_name, choices, scorer=fuzz.WRatio)
    if best and best[1] >= 70: return best[0]
    return tile_name

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
        chosen = None
        while i < len(events) and events[i].end < s_start: i += 1
        j = i; best_overlap = 0.0
        while j < len(events) and events[j].start <= s_end:
            ov = overlap((s_start, s_end), (events[j].start, events[j].end))
            if ov > best_overlap: best_overlap = ov; chosen = events[j]
            j += 1
        if chosen:
            mapped = map_tile_to_attendee_name(chosen.tile_name, attendees)
            out.append(TranscriptSegment(start=s_start, end=s_end, text=s_text, speaker=mapped, validated=chosen.validated, prob=s_prob))
        else:
            out.append(TranscriptSegment(start=s_start, end=s_end, text=s_text, speaker=None, validated=False, prob=s_prob))
    return out

# ----------------------------- Orchestrator -----------------------------
def build_active_speaker_timeline(video_path: Path, tiles: List[Tile], fps: float,
                                  video_workers: int, debug_dir: Optional[Path]) -> List[SpeakerEvent]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"Failed to open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    crop = CONFIG["VIDEO_CROP"]
    def crop_frame(f):
        if not crop: return f
        x,y,w,h = crop
        return f[y:y+h, x:x+w]

    step = int(max(1, round(native_fps / max(0.1, fps))))
    idx = 0
    jobs = []
    with ThreadPoolExecutor(max_workers=max(1, video_workers)) as ex:
        while True:
            ret = cap.grab()
            if not ret: break
            if idx % step == 0:
                ret, frame = cap.retrieve()
                if not ret or frame is None: break
                ts = idx / native_fps
                frame = crop_frame(frame)
                jobs.append(ex.submit(_detect_one_frame, ts, frame))
            idx += 1

        raw_events: List[SpeakerEvent] = []
        debug_counter = 0
        for fut in as_completed(jobs):
            ts, boxes, frame_dbg, mask_dbg = fut.result()
            matched = match_boxes_to_tiles(boxes, tiles)
            if matched is not None:
                raw_events.append(SpeakerEvent(start=ts, end=ts, tile_name=matched.name, validated=True))
            if CONFIG["DEBUG"] and debug_dir and frame_dbg is not None:
                if debug_counter % CONFIG["DEBUG_EVERY_N"] == 0:
                    canvas = frame_dbg.copy()
                    for t in tiles:
                        x,y,w,h = t.bbox; cv2.rectangle(canvas, (x,y), (x+w,y+h), (0,255,0), 2)
                    for (x,y,w,h) in boxes:
                        cv2.rectangle(canvas, (x,y), (x+w,y+h), (255,0,0), 2)
                    cv2.imwrite(str(debug_dir / f"frame_{int(ts*1000):08d}.jpg"), canvas)
                    if mask_dbg is not None:
                        cv2.imwrite(str(debug_dir / f"mask_{int(ts*1000):08d}.png"), mask_dbg)
                debug_counter += 1

    cap.release()

    if not raw_events: return []

    raw_events.sort(key=lambda e: e.start)
    merged: List[SpeakerEvent] = []
    cur = raw_events[0]
    merge_tol = (1.5 / max(0.1, fps))
    for ev in raw_events[1:]:
        if ev.tile_name == cur.tile_name and ev.start - cur.end <= merge_tol:
            cur.end = ev.end
        else:
            merged.append(cur)
            cur = ev
    merged.append(cur)

    # Smooth tiny gaps
    smoothed: List[SpeakerEvent] = []
    prev: Optional[SpeakerEvent] = None
    gap_tol = (1.0 / max(0.1, fps))
    for ev in merged:
        if prev and ev.start - prev.end < gap_tol:
            prev.end = ev.end
        else:
            if prev:
                smoothed.append(prev)
            prev = dataclasses.replace(ev)
    if prev:
        smoothed.append(prev)
    return smoothed

def _detect_one_frame(ts: float, frame: "np.ndarray"):
    boxes = detect_blue_border_regions(frame)
    return ts, boxes

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
    for pat, repl in VOCAB_RULES:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    return out

def map_tile_to_attendee_name(tile_name: str, attendees: Dict[str, str]) -> str:
    """Fuzzy map OCR'd tile name to a known attendee name from .ics."""
    if not attendees:
        return tile_name
    choices = list(attendees.keys())
    best = rf_process.extractOne(tile_name, choices, scorer=fuzz.WRatio)
    if best and best[1] >= 80:
        return best[0]
    return tile_name

def overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    s1, e1 = a
    s2, e2 = b
    s = max(s1, s2)
    e = min(e1, e2)
    return max(0.0, e - s)

def attribute_segments(segments: List[Dict], timeline: List[SpeakerEvent], attendees: Dict[str, str]) -> List[TranscriptSegment]:
    """Assign speaker to each STT segment by overlapping with active-speaker windows."""
    if not segments:
        return []
    events = sorted(timeline, key=lambda e: e.start)
    out: List[TranscriptSegment] = []
    i = 0
    for seg in segments:
        s_start, s_end, s_text = float(seg["start"]), float(seg["end"]), apply_vocab_rules(seg["text"])
        s_prob = seg.get("prob", None)
        chosen = None
        while i < len(events) and events[i].end < s_start:
            i += 1
        j = i
        best_overlap = 0.0
        while j < len(events) and events[j].start <= s_end:
            ov = overlap((s_start, s_end), (events[j].start, events[j].end))
            if ov > best_overlap:
                best_overlap = ov
                chosen = events[j]
            j += 1
        if chosen:
            mapped = map_tile_to_attendee_name(chosen.tile_name, attendees)
            out.append(TranscriptSegment(
                start=s_start, end=s_end, text=s_text, speaker=mapped, validated=chosen.validated, prob=s_prob
            ))
        else:
            out.append(TranscriptSegment(
                start=s_start, end=s_end, text=s_text, speaker=None, validated=False, prob=s_prob
            ))
    return out

# ----------------------------- Outputs -----------------------------
def write_vtt(segments: List[TranscriptSegment], out_path: Path, attendees: Dict[str, str]) -> None:
    out_lines = ["WEBVTT", ""]
    for i, seg in enumerate(segments, 1):
        start = hhmmss(seg.start)
        end = hhmmss(seg.end)
        speaker = seg.speaker or "UNKNOWN"
        company = attendees.get(seg.speaker or "", "")
        star = "*" if seg.validated and seg.speaker else ""
        speaker_line = f"{speaker}{star} ({company})" if company else f"{speaker}{star}"
        text = seg.text.strip()
        out_lines.append(str(i))
        out_lines.append(f"{start} --> {end}")
        out_lines.append(f"{speaker_line}: {text}")
        out_lines.append("")
    out_path.write_text("\n".join(out_lines), encoding="utf-8")

def write_segments_json(segments: List[TranscriptSegment], out_path: Path) -> None:
    data = [dataclasses.asdict(s) for s in segments]
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def write_qa_report(segments: List[TranscriptSegment], timeline: List[SpeakerEvent], out_path: Path) -> None:
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

# ----------------------------- Interactive Prompts -----------------------------
def _try_tk():
    try:
        import tkinter as _tk  # type: ignore
        from tkinter import filedialog as _fd  # type: ignore
        return _tk, _fd
    except Exception:
        return None, None

def _prompt_path_console(label: str, is_dir: bool = False) -> Path:
    print(f"[INPUT] {label} (drag the file/folder here then press Enter)")
    while True:
        raw = input("> ").strip().strip('"').strip("'")
        if not raw:
            print("Please provide a path.")
            continue
        path = Path(raw).expanduser().resolve()
        if is_dir and path.is_dir():
            return path
        if not is_dir and path.is_file():
            return path
        print(f"Path not valid as {'directory' if is_dir else 'file'}: {path}")

def _prompt_path_gui(label: str, is_dir: bool = False, filetypes=None) -> Optional[Path]:
    _tk, _fd = _try_tk()
    if not _tk:
        return None
    try:
        root = _tk.Tk()
        root.withdraw()
        path_str = ""
        if is_dir:
            path_str = _fd.askdirectory(title=label) or ""
        else:
            path_str = _fd.askopenfilename(title=label, filetypes=filetypes or [("All files", "*.*")]) or ""
        root.update_idletasks()
        root.destroy()
        if path_str:
            return Path(path_str).expanduser().resolve()
    except Exception:
        return None
    return None

def prompt_paths_interactive(args) -> Tuple[Path, Path, Path, Path]:
    video = args.video if isinstance(args.video, Path) else None
    screenshot = args.screenshot if isinstance(args.screenshot, Path) else None
    ics = args.ics if isinstance(args.ics, Path) else None
    outdir = args.outdir if isinstance(args.outdir, Path) else None

    if not video:
        video = _prompt_path_gui("Select meeting video (mp4/mkv/etc.)",
                                 filetypes=[("Video", "*.mp4 *.mkv *.mov *.avi"), ("All", "*.*")]) \
                or _prompt_path_console("Drop meeting video path")
    if not screenshot:
        screenshot = _prompt_path_gui("Select meeting grid screenshot (PNG/JPG)",
                                      filetypes=[("Images", "*.png *.jpg *.jpeg"), ("All", "*.*")]) \
                    or _prompt_path_console("Drop screenshot path")
    if not ics:
        ics = _prompt_path_gui("Select meeting .ics file", filetypes=[("ICS", "*.ics"), ("All", "*.*")]) \
              or _prompt_path_console("Drop .ics path")
    if not outdir:
        outdir = _prompt_path_gui("Select output directory", is_dir=True) \
                 or _prompt_path_console("Drop output directory", is_dir=True)

    return video, screenshot, ics, outdir

# ----------------------------- Orchestrator -----------------------------
def process_meeting(video: Path, screenshot: Path, ics: Path, outdir: Path,
                    stt_prefers_faster: bool, stt_model: str, fps: float,
                    device: str, compute_type: str, cpu_threads: int, whisper_workers: int,
                    language: Optional[str], tesseract_cmd: Optional[str],
                    video_workers: int) -> Dict[str, str]:
    ensure_deps()
    outdir.mkdir(parents=True, exist_ok=True)

    # CUDA sanity print (best-effort)
    try:
        import torch  # type: ignore
        print(f"[CUDA] torch.cuda.is_available() = {torch.cuda.is_available()}")
    except Exception:
        pass

    attendees = parse_ics_attendees(ics)

    name_bboxes = ocr_names_from_screenshot(screenshot, tesseract_cmd=tesseract_cmd)
    tiles = build_grid_from_names(name_bboxes)
    if not tiles:
        raise RuntimeError("No participant tiles recognized from screenshot OCR. Check image quality or OCR install.")

    timeline = build_active_speaker_timeline(video, tiles, fps=fps, video_workers=video_workers)

    wav_path = outdir / "audio_16k.wav"
    run_ffmpeg_extract_audio(video, wav_path)
    engine = stt_factory(prefer_faster=stt_prefers_faster, model_size=stt_model,
                         device=device, compute_type=compute_type,
                         cpu_threads=cpu_threads, num_workers=whisper_workers)
    print(f"[STT] engine={'faster-whisper' if isinstance(engine, FasterWhisperEngine) else 'openai-whisper'} "
          f"device={device} compute_type={compute_type} cpu_threads={cpu_threads} workers={whisper_workers} "
          f"lang={language or 'auto'}")
    stt_segments = engine.transcribe(wav_path, language=language)

    segments = attribute_segments(stt_segments, timeline, attendees)

    vtt_path = outdir / "speaker_attributed.vtt"
    json_path = outdir / "transcript_segments.json"
    qa_path = outdir / "qa_report.json"
    write_vtt(segments, vtt_path, attendees)
    write_segments_json(segments, json_path)
    write_qa_report(segments, timeline, qa_path)

    return {"vtt": str(vtt_path), "segments_json": str(json_path), "qa_report": str(qa_path)}

def main():
    p = argparse.ArgumentParser(description="Offline Teams meeting speaker attribution & transcription (V2-multithread)")
    # New flags
    p.add_argument("--interactive", action="store_true", help="Prompt for paths (GUI if available; console otherwise)")
    p.add_argument("--fps", type=float, default=2.0, help="Frame samples per second for border detection (default 2)")
    p.add_argument("--stt-model", type=str, default="medium", help="STT model size (small/base/medium/large)")
    p.add_argument("--prefer-faster", action="store_true", help="Prefer faster-whisper if available")
    p.add_argument("--device", default="auto", choices=["auto","cuda","cpu"], help="STT device")
    p.add_argument("--compute-type", default="float16", help="faster-whisper compute type (float16,int8,int8_float16, etc.)")
    p.add_argument("--cpu-threads", type=int, default=0, help="CPU threads for decoding (0=lib default)")
    p.add_argument("--whisper-workers", type=int, default=1, help="Parallel workers inside faster-whisper")
    p.add_argument("--lang", default="en", help="Force STT language (default: en). Use 'auto' to let the model detect.")
    p.add_argument("--video-workers", type=int, default=4, help="Threads for border detection compute")
    p.add_argument("--tesseract-cmd", type=str, default=None, help="Full path to tesseract.exe if not in PATH")

    # Optional positional args for paths (interactive will prompt if missing)
    p.add_argument("--video", type=Path, help="Path to meeting video (mp4/mkv/etc.)")
    p.add_argument("--screenshot", type=Path, help="High-res grid screenshot with names visible")
    p.add_argument("--ics", type=Path, help="Meeting .ics file with attendees")
    p.add_argument("--outdir", type=Path, help="Output directory")

    args = p.parse_args()

    # Interactive prompting if requested or any path is missing
    if args.interactive or not all([args.video, args.screenshot, args.ics, args.outdir]):
        args.video, args.screenshot, args.ics, args.outdir = prompt_paths_interactive(args)

    language = None if (args.lang or "").lower() == "auto" else args.lang

    try:
        outputs = process_meeting(
            video=args.video,
            screenshot=args.screenshot,
            ics=args.ics,
            outdir=args.outdir,
            stt_prefers_faster=args.prefer_faster,
            stt_model=args.stt_model,
            fps=args.fps,
            device=args.device,
            compute_type=args.compute_type,
            cpu_threads=args.cpu_threads,
            whisper_workers=args.whisper_workers,
            language=language,
            tesseract_cmd=args.tesseract_cmd,
            video_workers=args.video_workers,
        )
        print(json.dumps({"status": "ok", "outputs": outputs}, indent=2))
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
