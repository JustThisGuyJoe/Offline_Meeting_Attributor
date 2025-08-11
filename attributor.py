#!/usr/bin/env python3
"""
offline_meeting_attributor.py

Purpose:
    Offline pipeline to generate a speaker-attributed transcript from a Teams-style meeting video,
    using blue-border active speaker detection plus audio transcription. Designed for GPU-enabled
    Python environments and fully offline usage.

Key features:
    - Parses a .ics meeting invite to derive attendee names and (best-effort) company affiliations.
    - Parses a high-res meeting grid screenshot to build a tile map of participant names & positions.
    - Scans the meeting video for "blue border" rectangles around tiles to determine active speaker
      over time; robust to curved/wide monitors by allowing relaxed color/shape thresholds.
    - Transcribes audio locally (faster-whisper if available, else openai-whisper as fallback).
    - Aligns transcription segments to the active-speaker timeline to attribute lines to speakers.
    - Applies strict vocabulary normalization rules (ephOut, SATNO, Omitron, ephemeris) with
      whole-word & case-aware replacements.
    - Outputs:
        * speaker_attributed.vtt (WebVTT with speakers; "*" marks visually validated speakers)
        * transcript_segments.json (machine-readable segments with timing & speakers)
        * qa_report.json (attribution coverage & confidence metrics)

Design constraints:
    - DRY, SOLID-ish structure: small, testable classes; dependency injection for STT engine.
    - Clear naming, minimal docstrings; robust error messages (no silent failure).
    - Avoid deep nesting; prefer composition; keep optimization modest & safe.

CLI:
    python offline_meeting_attributor.py \
        --video path/to/meeting.mp4 \
        --screenshot path/to/grid.png \
        --ics path/to/meeting.ics \
        --outdir path/to/output \
        --fps 2

Dependencies (install as needed):
    pip install opencv-python numpy pillow rapidfuzz icalendar python-dateutil
    pip install pytesseract # requires Tesseract OCR installed on system
    pip install faster-whisper  # preferred (GPU)
    # or: pip install openai-whisper  # CPU/GPU depending on build

External tools:
    - ffmpeg in PATH (for audio track extraction if needed)
    - Tesseract OCR binary installed & in PATH (or set TESSERACT_CMD env)

Author: ChatGPT (offline-ready conversion for user's local stack)
License: MIT
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optional heavy deps — loaded lazily
try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    cv2 = None
try:
    import numpy as np  # type: ignore
except Exception as e:  # pragma: no cover
    np = None
try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    Image = None
try:
    import pytesseract  # type: ignore
except Exception as e:  # pragma: no cover
    pytesseract = None
try:
    from icalendar import Calendar  # type: ignore
except Exception as e:  # pragma: no cover
    Calendar = None
try:
    from rapidfuzz import fuzz, process as rf_process  # type: ignore
except Exception as e:  # pragma: no cover
    fuzz = None
    rf_process = None

# STT providers are injected via factory below.
class STTEngine:
    """Abstract STT engine interface."""
    def transcribe(self, audio_path: Path) -> List[Dict]:
        raise NotImplementedError


class FasterWhisperEngine(STTEngine):
    def __init__(self, model_size: str = "medium", device: Optional[str] = None):
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as e:
            raise RuntimeError("faster-whisper not installed. pip install faster-whisper") from e
        self.model = WhisperModel(model_size, device=device or "auto")

    def transcribe(self, audio_path: Path) -> List[Dict]:
        segments, _ = self.model.transcribe(str(audio_path), vad_filter=True)
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
    def __init__(self, model_size: str = "medium"):
        try:
            import whisper  # type: ignore
        except Exception as e:
            raise RuntimeError("openai-whisper not installed. pip install openai-whisper") from e
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path: Path) -> List[Dict]:
        import whisper  # type: ignore
        result = self.model.transcribe(str(audio_path), verbose=False)
        out = []
        for seg in result.get("segments", []):
            out.append({
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg["text"].strip(),
                "prob": float(seg.get("avg_logprob", 0.0)),
            })
        return out


def stt_factory(prefer_faster: bool = True, model_size: str = "medium") -> STTEngine:
    """Choose best available STT engine with clear errors otherwise."""
    if prefer_faster:
        try:
            return FasterWhisperEngine(model_size=model_size)
        except Exception:
            pass
    try:
        return OpenAIWhisperEngine(model_size=model_size)
    except Exception as e:
        raise RuntimeError(
            "No STT engine available. Install faster-whisper or openai-whisper."
        ) from e


# ----------------------------- Data Models -----------------------------

@dataclass
class Tile:
    name: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    grid_pos: Tuple[int, int]  # row, col


@dataclass
class SpeakerEvent:
    start: float
    end: float
    tile_name: str  # raw name from screenshot OCR
    validated: bool  # True if blue-border detected


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
    """Extract mono WAV via ffmpeg for STT."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video),
        "-ac", "1",
        "-ar", str(sample_rate),
        str(out_wav),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', errors='ignore')}")


def hhmmss(seconds: float) -> str:
    td = timedelta(seconds=float(max(0.0, seconds)))
    total_seconds = int(td.total_seconds())
    ms = int((seconds - int(seconds)) * 1000)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def ensure_deps():
    missing = []
    if cv2 is None:
        missing.append("opencv-python")
    if np is None:
        missing.append("numpy")
    if Image is None:
        missing.append("Pillow")
    if pytesseract is None:
        missing.append("pytesseract (+ Tesseract binary)")
    if Calendar is None:
        missing.append("icalendar")
    if fuzz is None or rf_process is None:
        missing.append("rapidfuzz")
    if missing:
        raise RuntimeError(f"Missing dependencies: {', '.join(missing)}")


# ----------------------------- ICS Parsing -----------------------------

def parse_ics_attendees(ics_path: Path) -> Dict[str, str]:
    """
    Parse attendee names from .ics. Returns dict mapping display-name -> company (best guess).

    Company inference heuristic:
      - If ORG or X-ORGANIZATION present, use it.
      - Else derive from email domain (e.g., jane@omitron.com -> Omitron).
    """
    attendees: Dict[str, str] = {}
    text = ics_path.read_bytes()
    cal = Calendar.from_ical(text)

    def domain_to_company(email: str) -> str:
        dom = email.split("@")[-1].split(">")[0].strip().lower()
        if "." in dom:
            base = dom.split(".")[0]
        else:
            base = dom
        # Normalize common company stylings
        mapping = {"omitron": "Omitron"}
        return mapping.get(base, base.capitalize())

    for comp in cal.walk():
        if comp.name == "VEVENT":
            # Organizer (optional)
            org = comp.get("ORGANIZER") or comp.get("organizer")
            if org:
                val = str(org)
                # ORGANIZER;CN=Name:mailto:email
                m = re.search(r"CN=([^:;]+)", val)
                cn = m.group(1).strip() if m else None
                email_m = re.search(r"mailto:([^>\s]+)", val, re.I)
                company = domain_to_company(email_m.group(1)) if email_m else ""
                if cn:
                    attendees[cn] = company or attendees.get(cn, "")

            # Attendees (icalendar stores as 'attendee' value or list; no .getall on Event)
            raw_atts = comp.get("attendee") or comp.get("ATTENDEE") or []
            if not isinstance(raw_atts, list):
                raw_atts = [raw_atts]

            for att in raw_atts:
                # Prefer params when available
                cn = None
                try:
                    params = getattr(att, "params", {})
                    if params and "CN" in params:
                        cn = str(params["CN"]).strip()
                except Exception:
                    pass
                sval = str(att)
                if not cn:
                    m = re.search(r"CN=([^:;]+)", sval)
                    cn = m.group(1).strip() if m else None
                email_m = re.search(r"mailto:([^>\s]+)", sval, re.I)
                company = domain_to_company(email_m.group(1)) if email_m else ""
                if cn:
                    attendees[cn] = company or attendees.get(cn, "")
    # Apply vocabulary rule normalization for Omitron spelling
    norm = {}
    for name, compy in attendees.items():
        compy = "Omitron" if compy.lower() == "omitron" else compy
        norm[name] = compy
    return norm


# ----------------------------- Screenshot OCR & Grid -----------------------------

def ocr_names_from_screenshot(screenshot_path: Path) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """
    Return list of (name, bbox) for tiles detected in the meeting screenshot.
    Heuristic:
      - Convert to grayscale, adaptive threshold to highlight text regions.
      - Use OpenCV contour detection to find name ribbons near tile bottoms.
      - Apply Tesseract OCR on these regions.
    """
    img = cv2.imread(str(screenshot_path))
    if img is None:
        raise RuntimeError(f"Failed to read screenshot: {screenshot_path}")
    h, w = img.shape[:2]

    # Preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 21, 10)

    # Find contours – candidate name areas (heuristic: wide + short near bottom of tiles)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / max(1, ch)
        area = cw * ch
        if area < 500 or aspect < 2.5:
            continue
        # Restrict to lower 70% of image (names usually bottom)
        if y < h * 0.3:
            continue
        candidates.append((x, y, cw, ch))

    # Merge overlapping candidates
    merged = []
    for x, y, cw, ch in sorted(candidates, key=lambda b: b[0]):
        if not merged:
            merged.append([x, y, x + cw, y + ch])
        else:
            mx1, my1, mx2, my2 = merged[-1]
            if x <= mx2 + 10 and y <= my2 + 10 and (x + cw) >= mx1 - 10:
                # Merge
                merged[-1] = [min(mx1, x), min(my1, y), max(mx2, x + cw), max(my2, y + ch)]
            else:
                merged.append([x, y, x + cw, y + ch])

    results = []
    for mx1, my1, mx2, my2 in merged:
        roi = img[my1:my2, mx1:mx2]
        if roi.size == 0:
            continue
        # OCR
        config = "--psm 7"
        text = pytesseract.image_to_string(roi, config=config)
        text = re.sub(r"[\r\n]+", " ", text).strip()
        if not text:
            continue
        results.append((text, (mx1, my1, mx2 - mx1, my2 - my1)))
    return results


def build_grid_from_names(name_bboxes: List[Tuple[str, Tuple[int, int, int, int]]]) -> List[Tile]:
    """Assign grid positions to tiles by sorting by y then x; name normalization applied."""
    if not name_bboxes:
        return []
    # Sort by y then x to approximate grid placement
    name_bboxes_sorted = sorted(name_bboxes, key=lambda x: (x[1][1], x[1][0]))
    rows: List[List[Tuple[str, Tuple[int, int, int, int]]]] = []
    row_threshold = 50  # vertical clustering tolerance
    for name, bbox in name_bboxes_sorted:
        x, y, w, h = bbox
        if not rows:
            rows.append([(name, bbox)])
            continue
        last_row_y = np.mean([b[1][1] for b in rows[-1]])
        if abs(y - last_row_y) <= row_threshold:
            rows[-1].append((name, bbox))
        else:
            rows.append([(name, bbox)])
    # Assign grid positions
    tiles: List[Tile] = []
    for r_idx, row in enumerate(rows):
        row_sorted = sorted(row, key=lambda x: x[1][0])
        for c_idx, (name, bbox) in enumerate(row_sorted):
            clean = " ".join(name.split())
            tiles.append(Tile(name=clean, bbox=bbox, grid_pos=(r_idx, c_idx)))
    return tiles


# ----------------------------- Blue Border Detection -----------------------------

def detect_blue_border_regions(frame: "np.ndarray") -> List[Tuple[int, int, int, int]]:
    """
    Detect blue-ish rectangular borders in a frame. Returns bounding boxes.
    Approach:
      - Convert to HSV; threshold for blue hue range (tolerant/curved screens).
      - Morph close regions; find contours; keep rectangular-ish shapes with thin strokes.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Liberal blue ranges (two ranges to cover wrap-around)
    lower1 = np.array([90, 50, 50])
    upper1 = np.array([130, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)

    # Morph to connect edges
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        perim = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if w < 40 or h < 40:
            continue
        if area < 200:
            continue
        # Prefer elongated border rectangles (thin-ish)
        stroke_ratio = area / max(1.0, w * h)
        if stroke_ratio > 0.25:
            continue
        boxes.append((x, y, w, h))
    return boxes


def match_boxes_to_tiles(boxes: List[Tuple[int, int, int, int]], tiles: List[Tile]) -> Optional[Tile]:
    """Choose the tile whose bbox overlaps the blue border box the most (IoU)."""
    if not boxes or not tiles:
        return None

    def iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        a2 = (ax + aw, ay + ah)
        b2 = (bx + bw, by + bh)
        x_left = max(ax, bx)
        y_top = max(ay, by)
        x_right = min(a2[0], b2[0])
        y_bottom = min(a2[1], b2[1])
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        inter = (x_right - x_left) * (y_bottom - y_top)
        union = aw * ah + bw * bh - inter
        return inter / max(1e-6, union)

    best = None
    best_iou = 0.0
    for b in boxes:
        for t in tiles:
            i = iou(b, t.bbox)
            if i > best_iou:
                best_iou = i
                best = t
    # Require minimal overlap confidence
    return best if best_iou >= 0.05 else None


def build_active_speaker_timeline(video_path: Path, tiles: List[Tile], fps: float = 2.0) -> List[SpeakerEvent]:
    """
    Sample frames at FPS; detect blue border; map to tile; return merged timeline intervals.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / max(1.0, native_fps)

    step = int(max(1, round(native_fps / max(0.1, fps))))
    idx = 0
    time_s = 0.0
    raw_events: List[SpeakerEvent] = []

    while True:
        ret = cap.grab()
        if not ret:
            break
        if idx % step == 0:
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                break
            time_s = idx / native_fps
            boxes = detect_blue_border_regions(frame)
            matched = match_boxes_to_tiles(boxes, tiles)
            if matched is not None:
                raw_events.append(SpeakerEvent(
                    start=time_s, end=time_s, tile_name=matched.name, validated=True
                ))
        idx += 1

    cap.release()

    # Merge contiguous events for same tile
    merged: List[SpeakerEvent] = []
    if not raw_events:
        return merged

    cur = raw_events[0]
    for ev in raw_events[1:]:
        if ev.tile_name == cur.tile_name and ev.start - cur.end <= (1.5 / fps):
            cur.end = ev.end
        else:
            merged.append(cur)
            cur = ev
    merged.append(cur)
    # Smooth small gaps by extending neighbors if close
    smoothed: List[SpeakerEvent] = []
    prev: Optional[SpeakerEvent] = None
    for ev in merged:
        if prev and ev.start - prev.end < (1.0 / fps):
            prev.end = ev.end
        else:
            if prev:
                smoothed.append(prev)
            prev = dataclasses.replace(ev)
    if prev:
        smoothed.append(prev)
    return smoothed


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
    # RapidFuzz ratio mapping; fallback to raw tile if low score
    best = rf_process.extractOne(tile_name, choices, scorer=fuzz.WRatio)
    if best and best[1] >= 80:
        return best[0]
    return tile_name


def attribute_segments(segments: List[Dict], timeline: List[SpeakerEvent], attendees: Dict[str, str]) -> List[TranscriptSegment]:
    """
    Assign speaker to each STT segment by overlapping time with active-speaker windows.
    """
    if not segments:
        return []
    events = sorted(timeline, key=lambda e: e.start)
    out: List[TranscriptSegment] = []
    i = 0
    for seg in segments:
        s_start, s_end, s_text = float(seg["start"]), float(seg["end"]), apply_vocab_rules(seg["text"])
        s_prob = seg.get("prob", None)
        # Find overlapping event
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
            # Map OCR tile name to canonical attendee name (if matched)
            mapped = map_tile_to_attendee_name(chosen.tile_name, attendees)
            out.append(TranscriptSegment(
                start=s_start, end=s_end, text=s_text, speaker=mapped, validated=chosen.validated, prob=s_prob
            ))
        else:
            out.append(TranscriptSegment(
                start=s_start, end=s_end, text=s_text, speaker=None, validated=False, prob=s_prob
            ))
    return out


def overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    s1, e1 = a
    s2, e2 = b
    s = max(s1, s2)
    e = min(e1, e2)
    return max(0.0, e - s)


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
    print(f"[INPUT] {label} (you can drag the file/folder into this window and press Enter)")
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
    """
    Prompt user to select video, screenshot, ics, and output directory.
    Supports GUI via Tkinter if available; else console prompt that accepts drag-and-drop.
    """
    video = args.video if isinstance(args.video, Path) else None
    screenshot = args.screenshot if isinstance(args.screenshot, Path) else None
    ics = args.ics if isinstance(args.ics, Path) else None
    outdir = args.outdir if isinstance(args.outdir, Path) else None

    # Video
    if not video:
        video = _prompt_path_gui("Select meeting video (mp4/mkv/etc.)",
                                 filetypes=[("Video", "*.mp4 *.mkv *.mov *.avi"), ("All", "*.*")]) \
                or _prompt_path_console("Drop meeting video path")
    # Screenshot
    if not screenshot:
        screenshot = _prompt_path_gui("Select meeting grid screenshot (PNG/JPG)",
                                      filetypes=[("Images", "*.png *.jpg *.jpeg"), ("All", "*.*")]) \
                    or _prompt_path_console("Drop screenshot path")
    # ICS
    if not ics:
        ics = _prompt_path_gui("Select meeting .ics file", filetypes=[("ICS", "*.ics"), ("All", "*.*")]) \
              or _prompt_path_console("Drop .ics path")
    # Outdir
    if not outdir:
        outdir = _prompt_path_gui("Select output directory", is_dir=True) \
                 or _prompt_path_console("Drop output directory", is_dir=True)

    return video, screenshot, ics, outdir

# ----------------------------- Orchestrator -----------------------------

def process_meeting(video: Path, screenshot: Path, ics: Path, outdir: Path, stt_prefers_faster: bool, stt_model: str, fps: float) -> Dict[str, str]:
    ensure_deps()
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) ICS
    attendees = parse_ics_attendees(ics)

    # 2) Screenshot -> tiles
    name_bboxes = ocr_names_from_screenshot(screenshot)
    tiles = build_grid_from_names(name_bboxes)
    if not tiles:
        raise RuntimeError("No participant tiles recognized from screenshot OCR. Check image quality or OCR install.")

    # 3) Video -> active speaker timeline
    timeline = build_active_speaker_timeline(video, tiles, fps=fps)

    # 4) Audio -> STT
    wav_path = outdir / "audio_16k.wav"
    run_ffmpeg_extract_audio(video, wav_path)
    engine = stt_factory(prefer_faster=stt_prefers_faster, model_size=stt_model)
    stt_segments = engine.transcribe(wav_path)

    # 5) Align & attribute
    segments = attribute_segments(stt_segments, timeline, attendees)

    # 6) Outputs
    vtt_path = outdir / "speaker_attributed.vtt"
    json_path = outdir / "transcript_segments.json"
    qa_path = outdir / "qa_report.json"
    write_vtt(segments, vtt_path, attendees)
    write_segments_json(segments, json_path)
    write_qa_report(segments, timeline, qa_path)

    return {
        "vtt": str(vtt_path),
        "segments_json": str(json_path),
        "qa_report": str(qa_path),
    }


def main():
    p = argparse.ArgumentParser(description="Offline Teams meeting speaker attribution & transcription")
    p.add_argument("--interactive", action="store_true", help="Prompt for paths (GUI if available; console otherwise)")
    p.add_argument("--video", type=Path, required=True, help="Path to meeting video (mp4/mkv/etc.)")
    p.add_argument("--screenshot", type=Path, required=True, help="High-res grid screenshot with names visible")
    p.add_argument("--ics", type=Path, required=True, help="Meeting .ics file with attendees")
    p.add_argument("--outdir", type=Path, required=True, help="Output directory")
    p.add_argument("--fps", type=float, default=2.0, help="Frame samples per second for border detection (default 2)")
    p.add_argument("--stt-model", type=str, default="medium", help="STT model size (small/base/medium/large)")
    p.add_argument("--prefer-faster", action="store_true", help="Prefer faster-whisper if available")
    args = p.parse_args()

    # If interactive or any required path is missing, prompt one-by-one
    if args.interactive or not all([args.video, args.screenshot, args.ics, args.outdir]):
        args.video, args.screenshot, args.ics, args.outdir = prompt_paths_interactive(args)

    try:
        outputs = process_meeting(
            video=args.video,
            screenshot=args.screenshot,
            ics=args.ics,
            outdir=args.outdir,
            stt_prefers_faster=args.prefer_faster,
            stt_model=args.stt_model,
            fps=args.fps,
        )
        print(json.dumps({"status": "ok", "outputs": outputs}, indent=2))
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
