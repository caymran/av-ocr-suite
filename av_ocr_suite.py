#!/usr/bin/env python3
"""
AV + OCR Suite (v2.3) — Live Audio Transcription, Screen OCR, and Auto-Archive
Author: Caymran Cummings (2025)
File: av_ocr_suite.py

Now using Faster-Whisper (CTranslate2) on CPU with int8 for small memory + speed.
Downloads models into an app cache directory; subsequent runs reuse them.


AV + OCR Suite (v2.2) — Live Audio Transcription, Screen OCR, and Auto-Archive
Author: Caymran Cummings (2025)
File: av_ocr_suite.py

Changes in v2.2 (compared to v2.1)
----------------------------------
• Decoupled core behavior from UI buttons:
  - Audio recording auto-starts as soon as the input stream opens (no hidden timer).
  - Screen capture/OCR emits ONLY after a window is selected (default black is idle).
  - Close (window X) always stops, flushes, exports (MP4 if frames, else MP3), then cleans temps.
• Hardened close/export guard:
  - Latch temp WAV if last path wasn’t captured.
  - Force final transcription, write transcripts, export, then cleanup temps.
• Frame capture logic:
  - Save frames only when an audio timeline exists; avoids orphan/duplicate frames.

Overview
--------
One GUI that:
  • Captures live audio and transcribes it with Whisper.
  • Captures periodic screen frames and OCRs them with Tesseract.
  • Aligns frames on the *audio timeline* and, on exit, exports either:
      - a compiled MP4 (video + audio) if any frames were captured, or
      - an MP3 (audio-only) otherwise.
  • Writes human-readable audio and OCR transcripts.
  • Produces a clean, repeatable archive layout, then removes all temporary files.

Run Layout (created at startup)
-------------------------------
CWD/
  logs/
    app_YYYYMMDD_HHMMSS.log          # Execution log (this run)
    stdout_YYYYMMDD_HHMMSS.log       # stdout/stderr redirection (frozen builds)
  archives/
    YYYYMMDD_HHMMSS/                 # One subfolder per run (the "run dir")
      _temp/                         # Ephemeral working area (deleted on exit)
        frames/                      # Staged JPEG frames + ffconcat during build
        audio_YYYYMMDD_HHMMSS.wav    # Temporary WAV while recording

Final Artifacts (written on exit)
---------------------------------
Inside: CWD/archives/YYYYMMDD_HHMMSS/
  • YYYYMMDD_HHMMSS_ocr_transcript_<meetingname>.txt
  • YYYYMMDD_HHMMSS_audio_transcript_<meetingname>.txt
  • If frames captured:  YYYYMMDD_HHMMSS_video_<meetingname>.mp4
    Else (audio-only):   YYYYMMDD_HHMMSS_audio_<meetingname>.mp3

Naming Rules
------------
- date_time:  YYYYMMDD_HHMMSS (UTC/local per system clock at app start).
- meetingname: best-effort guess from the picked window title, sanitized:
    [A-Z a-z 0-9 _] only; spaces/dashes → underscores; trimmed; fallback "meeting".
- custom_prompt.txt must live next to this script (SCRIPT_DIR/custom_prompt.txt).
  If missing, it is created with a default prompt.

Behavioral Guarantees
---------------------
- No stray files in CWD besides `logs/` and `archives/`.
- All *working* files live under `archives/<run>/_temp/` during execution.
- `_temp/` is fully removed on successful shutdown; final artifacts remain.
- Video export uses ffmpeg concat with still-image tuning and audio-clocked durations.

Key Dependencies
----------------
- Python 3.10+ recommended.
- Whisper (openai/whisper), PyTorch, pydub, librosa, webrtcvad, numpy.
- PyQt5 for GUI; Tesseract + pytesseract for OCR (optional but recommended).
- ffmpeg must be available in PATH for media export.

Quick Start
-----------
1) Ensure `ffmpeg` and (optionally) Tesseract are installed and on PATH.
2) Place `custom_prompt.txt` next to this script (auto-created if absent).
3) `python av_ocr_suite.py`
4) Pick an audio input device; optionally select a window to OCR.
5) Close the app to trigger export + cleanup. Final outputs are in `archives/<run>/`.

Notes
-----
- Uses audio-timeline stamping: each frame is time-aligned to the current audio time.
- GUI provides combined "Copy Transcript" (Audio + OCR with a structured header).
- Designed for PyInstaller; includes DPI awareness and windowing niceties on Windows.
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import wave
import shutil
import ctypes
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from threading import Thread, Event, Lock
from collections import deque
from contextlib import contextmanager

# --- keep warnings hush for frozen builds etc. ---
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=r"pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r"TypedStorage is deprecated")

# Third-party
import numpy as np
import webrtcvad
import pyperclip
import pyaudiowpatch as pyaudio

# --- ensure PyInstaller bundles OCR deps (no-op if missing) ---
try:
    import pytesseract  # noqa: F401
    from PIL import Image  # noqa: F401
except Exception:
    pass


# Qt
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton, QTabWidget, QMessageBox
from PyQt5.QtCore import Qt

# Logging
import logging
from logging.handlers import RotatingFileHandler

DEBUG_MODE = '--debug' in sys.argv

# ======================
#   Paths & Naming
# ======================
APP_START = datetime.now()
STAMP = APP_START.strftime("%Y%m%d_%H%M%S")

SCRIPT_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) if getattr(sys, "frozen", False) \
    else Path(__file__).resolve().parent

# Where the executable lives when frozen; else the script dir
EXE_DIR = Path(getattr(sys, "frozen", False) and sys.executable or __file__).resolve().parent

CWD = Path.cwd()
ROOT_LOGS = CWD / "logs"
ROOT_ARCHIVES = CWD / "archives"
RUN_DIR = ROOT_ARCHIVES / STAMP
TEMP_DIR = RUN_DIR / "_temp"
TEMP_FRAMES = TEMP_DIR / "frames"
TEMP_WAV = TEMP_DIR / f"audio_{STAMP}.wav"
TEMP_MP3_DEBUG = TEMP_DIR / f"debug_{STAMP}.mp3"

# App cache for model downloads (persists across runs/builds; small + fast)
APP_CACHE = (Path(os.getenv("LOCALAPPDATA", Path.home())) / "AV_OCR_Suite" / "models_cache")

def _frozen_base_dir() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))

def _enable_tesseract_if_present():
    """Wire up pytesseract if Tesseract is installed or bundled, and add its dir to PATH."""
    try:
        import pytesseract
        from shutil import which

        # 1) PATH
        tcmd = which("tesseract")

        # 2) Bundled locations
        candidates = [
            EXE_DIR / "tesseract" / "tesseract.exe",
            CWD / "tesseract" / "tesseract.exe",
            Path(os.getenv("LOCALAPPDATA", "")) / "AV_OCR_Suite" / "tesseract" / "tesseract.exe",
        ]
        if not tcmd:
            for c in candidates:
                if c and c.exists():
                    tcmd = str(c)
                    break

        if tcmd:
            tdir = str(Path(tcmd).parent)
            # Prepend so our bundled copy wins
            os.environ["PATH"] = tdir + os.pathsep + os.environ.get("PATH", "")
            pytesseract.pytesseract.tesseract_cmd = tcmd
            global USE_TESS
            USE_TESS = True
            log.info(f"Tesseract enabled: {tcmd}")
        else:
            log.info("Tesseract not found; OCR disabled.")
    except Exception:
        log.exception("Failed to initialize Tesseract; OCR disabled.")

# meeting name helpers
def sanitize_meeting_name(name: str) -> str:
    cleaned = []
    for ch in name.replace(" ", "_").replace("-", "_"):
        if ch.isalnum() or ch == "_":
            cleaned.append(ch)
    cleaned = "".join(cleaned).strip("_")
    return cleaned or "meeting"

MEETING_NAME = sanitize_meeting_name("meeting")

def ensure_dirs():
    ROOT_LOGS.mkdir(parents=True, exist_ok=True)
    ROOT_ARCHIVES.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_FRAMES.mkdir(parents=True, exist_ok=True)
    APP_CACHE.mkdir(parents=True, exist_ok=True)

ensure_dirs()

# Final artifact paths
def ocr_txt_path(meeting: str) -> Path: return RUN_DIR / f"{STAMP}_ocr_transcript_{sanitize_meeting_name(meeting)}.txt"
def audio_txt_path(meeting: str) -> Path: return RUN_DIR / f"{STAMP}_audio_transcript_{sanitize_meeting_name(meeting)}.txt"
def video_mp4_path(meeting: str) -> Path: return RUN_DIR / f"{STAMP}_video_{sanitize_meeting_name(meeting)}.mp4"
def audio_mp3_path(meeting: str) -> Path: return RUN_DIR / f"{STAMP}_audio_{sanitize_meeting_name(meeting)}.mp3"

# ======================
#   Audio Config
# ======================
CHUNK_DURATION_MS = 30
FORMAT = pyaudio.paInt16
SAMPLE_WIDTH = 2
VAD_MODE = 1
BUFFER_SECONDS = 30
CHUNKS_PER_SECOND = 1000 // CHUNK_DURATION_MS
POOL_SECONDS = 7
POOL_CHUNKS = CHUNKS_PER_SECOND * POOL_SECONDS

# custom_prompt.txt
CUSTOM_PROMPT_FILE = SCRIPT_DIR / "custom_prompt.txt"
DEFAULT_PROMPT = (
    "When addressing the following prompt - keep in mind - text created from transcription is best guess "
    "phonetically by whisper AI. it does make mistakes. The OCR transcript is best guess by tesseract. "
    "If things look like other letters, it will make its best guess. Assume the provided transcripts may "
    "have errors or misinterpretations. Make best guesses as to corrections based on a summation of all provided data. "
    "OCR participant names are likely more correct than audio transcribed. Make substitutions as needed. "
    "Use OCR to correct confusing audio phrases. Generate meeting minutes titled by the best-guess topic, "
    "include start time [{start}], duration [{duration}], Speakers, Summary, Topics Covered, Q&A, and Action Items. "
    "Write casually as if a regular attendee, not as an AI. Avoid timestamp references. "
    "If unsure of names, list them and ask for clarification in the next prompt."
)

# ======================
#        Logging
# ======================
def setup_logging():
    logger = logging.getLogger("transcriber")
    logger.setLevel(logging.INFO)
    if sys.stdout:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(console)
    logfile = ROOT_LOGS / f"app_{STAMP}.log"
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    logger.info(f"Application log: {logfile}")
    return logger

log = setup_logging()

def safe_flush():
    try:
        if sys.stdout:
            sys.stdout.flush()
    except Exception:
        pass

def ensure_custom_prompt_file():
    try:
        if CUSTOM_PROMPT_FILE.exists():
            return CUSTOM_PROMPT_FILE.read_text(encoding="utf-8")
        else:
            CUSTOM_PROMPT_FILE.write_text(DEFAULT_PROMPT, encoding="utf-8")
            return DEFAULT_PROMPT
    except Exception:
        log.exception("Error ensuring custom prompt file; using default.")
        return DEFAULT_PROMPT

custom_header = ensure_custom_prompt_file()

# ======================
#   FFmpeg helpers + Export
# ======================
def _ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _resample_i16_for_vad(mono_i16: np.ndarray, src_sr: int, dst_sr: int) -> bytes:
    """Quick/cheap linear resample int16 -> int16 for short VAD frames."""
    if src_sr == dst_sr:
        return mono_i16.tobytes()
    x = mono_i16.astype(np.float32) / 32768.0
    dur = x.size / float(src_sr)
    if dur <= 0.0:
        return b""
    n_new = max(1, int(round(dur * dst_sr)))
    t_old = np.linspace(0.0, dur, num=x.size, endpoint=False, dtype=np.float32)
    t_new = np.linspace(0.0, dur, num=n_new,   endpoint=False, dtype=np.float32)
    y = np.interp(t_new, t_old, x)
    y16 = np.clip(np.round(y * 32768.0), -32768, 32767).astype(np.int16)
    return y16.tobytes()


def export_archive_build_video(frames_with_ts, wav_path=None, crf=28, output_path=None):
    """
    frames_with_ts: list of tuples (path, t_seconds) sorted by t_seconds (already on audio clock)
    wav_path: optional wav to mux
    output_path: Path object or str for final .mp4
    """
    log = logging.getLogger("transcriber")
    if not frames_with_ts:
        raise RuntimeError("export: no frames provided")

    # Ensure timestamps strictly increase and compute per-frame durations
    frames = sorted(frames_with_ts, key=lambda x: float(x[1]))
    times = [float(t) for _, t in frames]
    # Clamp non-decreasing + compute durations
    for i in range(1, len(times)):
        if times[i] < times[i-1]:
            times[i] = times[i-1]
    durations = []
    for i in range(len(times)):
        if i < len(times) - 1:
            d = max(0.04, times[i+1] - times[i])   # minimum ~1/25s
        else:
            d = 1.0                                 # tail default if we don’t know true end
        durations.append(d)

    # Build ffconcat file beside frames
    first_dir = Path(frames[0][0]).parent
    concat_file = first_dir / "frames.ffconcat"
    with open(concat_file, "w", encoding="utf-8") as f:
        f.write("ffconcat version 1.0\n")
        for (img, _), dur in zip(frames, durations):
            # Use relative paths (safer with -safe 0, but keep it simple)
            f.write(f"file {Path(img).name}\n")
            f.write(f"duration {dur:.6f}\n")

    # If the last frame needs duration respected, ffmpeg requires repeating the last file once
    f_last = Path(frames[-1][0]).name
    with open(concat_file, "a", encoding="utf-8") as f:
        f.write(f"file {f_last}\n")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build ffmpeg command
    cmd = [
        "ffmpeg", "-y",
        "-safe", "0",
        "-f", "concat", "-i", str(concat_file),
    ]
    if wav_path:
        cmd += ["-i", str(wav_path)]

    # 🔧 Ensure even dimensions + proper colorspace/SAR
    vf = "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p,setsar=1"

    cmd += [
        "-vf", vf,            # <— add this line
        "-r", "25",
        "-c:v", "libx264", "-crf", str(int(crf)),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]
    if wav_path:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        # No audio input, but keep aac track? we’ll skip audio entirely
        pass
    cmd += [str(out)]

    logger = logging.getLogger("transcriber")
    logger.info(f"[export] ffmpeg cmd: {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.stdout:
        logger.info(f"[ffmpeg] stdout:\n{proc.stdout}")
    if proc.stderr:
        logger.info(f"[ffmpeg] stderr:\n{proc.stderr}")

    if proc.returncode != 0 or not out.exists() or out.stat().st_size < 1024:
        raise RuntimeError(f"ffmpeg concat export failed (code {proc.returncode})")

TARGET_SR = 16000

# --- helpers for dual-source tagging and interleaving ---
def interleave_tagged_segments(mic_segments, spk_segments):
    tagged = []
    for s in mic_segments:
        if getattr(s, "text", "").strip():
            tagged.append(("MIC", float(getattr(s, "start", 0.0)), s.text.strip()))
    for s in spk_segments:
        if getattr(s, "text", "").strip():
            tagged.append(("SPK", float(getattr(s, "start", 0.0)), s.text.strip()))
    tagged.sort(key=lambda t: t[1])
    return [f"[{src}] {txt}" for (src, _, txt) in tagged]

def i16_bytes_to_f32_16k(b, sr):
    if not b:
        return np.zeros(0, dtype=np.float32)
    x = np.frombuffer(b, dtype=np.int16)
    if x.size == 0:
        return np.zeros(0, dtype=np.float32)
    mono = x.astype(np.float32) / 32768.0
    if int(sr) == 16000:
        return mono
    return resample_to_16k(mono, int(sr))

def resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if x is None or x.size == 0:
        return np.zeros(0, dtype=np.float32)
    sr = int(max(1, sr))
    if sr == TARGET_SR:
        return x.astype(np.float32, copy=False)
    n = x.size
    dur = n / float(sr)
    if dur <= 0.0:
        return np.zeros(0, dtype=np.float32)
    t_old = np.linspace(0.0, dur, num=n, endpoint=False, dtype=np.float64)
    n_new = max(2, int(round(dur * TARGET_SR)))
    t_new = np.linspace(0.0, dur, num=n_new, endpoint=False, dtype=np.float64)
    y = np.interp(t_new, t_old, x.astype(np.float64, copy=False))
    return y.astype(np.float32, copy=False)

def _resample_i16_for_vad(mono_i16: np.ndarray, src_sr: int, dst_sr: int = 16000) -> bytes:
    """
    Cheap linear resample int16 -> int16 for short VAD frames.
    Returns bytes suitable for webrtcvad (10/20/30 ms @ 8k/16k/32k/48k).
    """
    if src_sr == dst_sr:
        return mono_i16.tobytes()
    # normalize to float
    x = mono_i16.astype(np.float32) / 32768.0
    dur = x.size / float(src_sr)
    if dur <= 0.0:
        return b""
    n_new = max(1, int(round(dur * dst_sr)))
    t_old = np.linspace(0.0, dur, num=x.size, endpoint=False)
    t_new = np.linspace(0.0, dur, num=n_new, endpoint=False)
    y = np.interp(t_new, t_old, x)
    y16 = np.clip(np.round(y * 32768.0), -32768, 32767).astype(np.int16)
    return y16.tobytes()

# Force Selector event loop on Windows to avoid frozen-app asyncio issues.
import asyncio
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

def _ensure_ffmpeg_on_path():
    """Ensure bundled ffmpeg folder is added to PATH if ffmpeg isn't already found."""
    import shutil
    if shutil.which("ffmpeg"):
        return
    candidates = [
        EXE_DIR / "ffmpeg",
        CWD / "ffmpeg",
        Path(os.getenv("LOCALAPPDATA", "")) / "AV_OCR_Suite" / "ffmpeg",
    ]
    for d in candidates:
        if d.exists() and (d / "ffmpeg.exe").exists():
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            log.info(f"ffmpeg enabled from: {d}")
            return
    log.info("ffmpeg not found on PATH and no bundled copy detected.")


# ======================
#   OCR helpers + Win32 capture
# ======================
USE_TESS = False
fwhisper_model = None  # faster-whisper model handle

def qimage_to_gray_np(img: QtGui.QImage) -> np.ndarray:
    qg = img.convertToFormat(QtGui.QImage.Format.Format_Grayscale8)
    w = qg.width(); h = qg.height()
    if w <= 0 or h <= 0:
        return np.empty((0, 0), dtype=np.uint8)
    bpl = qg.bytesPerLine()
    ptr = qg.bits(); ptr.setsize(bpl * h)
    arr_padded = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl))
    return arr_padded[:, :w].copy()

def qimage_is_mostly_black(img: QtGui.QImage, sample=1500, threshold=16) -> bool:
    if img.isNull():
        return True
    gray = qimage_to_gray_np(img)
    h, w = gray.shape
    if h == 0 or w == 0:
        return True
    n = min(sample, gray.size)
    if n <= 0:
        return True
    rng = np.random.default_rng()
    ys = rng.integers(0, h, size=n, endpoint=False)
    xs = rng.integers(0, w, size=n, endpoint=False)
    samples = gray[ys, xs]
    return (samples < threshold).mean() >= 0.95

# --- Win32 capture bits (unchanged) ---
from ctypes import wintypes
user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32
dwmapi = ctypes.WinDLL("dwmapi")

PW_RENDERFULLCONTENT = 0x00000002
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79
GA_ROOT = 2

def get_window_title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value

def get_class_name(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value

def get_pid(hwnd: int) -> int:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value

def _get_extended_frame_bounds(hwnd: int) -> wintypes.RECT:
    DWMWA_EXTENDED_FRAME_BOUNDS = 9
    rect = wintypes.RECT()
    hr = dwmapi.DwmGetWindowAttribute(
        wintypes.HWND(hwnd),
        ctypes.c_uint(DWMWA_EXTENDED_FRAME_BOUNDS),
        ctypes.byref(rect),
        ctypes.sizeof(rect),
    )
    if hr == 0:
        return rect
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise OSError("GetWindowRect failed")
    return rect

def get_window_rect(hwnd: int):
    rect = _get_extended_frame_bounds(hwnd)
    return rect.left, rect.top, rect.right, rect.bottom

def _extract_qimage_from_dc(mem_dc, bmp, w, h) -> QtGui.QImage:
    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ('biSize', wintypes.DWORD), ('biWidth', wintypes.LONG),
            ('biHeight', wintypes.LONG), ('biPlanes', wintypes.WORD),
            ('biBitCount', wintypes.WORD), ('biCompression', wintypes.DWORD),
            ('biSizeImage', wintypes.DWORD), ('biXPelsPerMeter', wintypes.LONG),
            ('biYPelsPerMeter', wintypes.LONG), ('biClrUsed', wintypes.DWORD),
            ('biClrImportant', wintypes.DWORD),
        ]
    class BITMAPINFO(ctypes.Structure):
        _fields_ = [('bmiHeader', BITMAPINFOHEADER), ('bmiColors', wintypes.DWORD * 3)]
    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = w
    bmi.bmiHeader.biHeight = -h
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = 0
    buf_len = w * h * 4
    bits = (ctypes.c_ubyte * buf_len)()
    got = gdi32.GetDIBits(mem_dc, bmp, 0, h, ctypes.byref(bits), ctypes.byref(bmi), 0)
    if got != h:
        raise OSError("GetDIBits failed")
    img = QtGui.QImage(bytes(bits), w, h, w * 4, QtGui.QImage.Format.Format_RGB32)
    return img.copy()

def _capture_via_printwindow_or_bitblt(hwnd: int, left: int, top: int, w: int, h: int) -> QtGui.QImage:
    hwnd_dc = user32.GetWindowDC(hwnd)
    if not hwnd_dc:
        raise OSError("GetWindowDC failed")
    mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
    bmp = gdi32.CreateCompatibleBitmap(hwnd_dc, w, h)
    gdi32.SelectObject(mem_dc, bmp)
    ok = user32.PrintWindow(hwnd, mem_dc, PW_RENDERFULLCONTENT)
    if ok:
        img = _extract_qimage_from_dc(mem_dc, bmp, w, h)
        if not qimage_is_mostly_black(img):
            gdi32.DeleteObject(bmp); gdi32.DeleteDC(mem_dc); user32.ReleaseDC(hwnd, hwnd_dc)
            return img
    screen_dc = user32.GetDC(0)
    if not screen_dc:
        gdi32.DeleteObject(bmp); gdi32.DeleteDC(mem_dc); user32.ReleaseDC(hwnd, hwnd_dc)
        raise OSError("GetDC(0) failed")
    vx = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    vy = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    src_x = left - vx
    src_y = top - vy
    SRCCOPY = 0x00CC0020
    if not gdi32.BitBlt(mem_dc, 0, 0, w, h, screen_dc, src_x, src_y, SRCCOPY):
        user32.ReleaseDC(0, screen_dc)
        gdi32.DeleteObject(bmp); gdi32.DeleteDC(mem_dc); user32.ReleaseDC(hwnd, hwnd_dc)
        raise OSError("BitBlt fallback failed")
    user32.ReleaseDC(0, screen_dc)
    img = _extract_qimage_from_dc(mem_dc, bmp, w, h)
    gdi32.DeleteObject(bmp); gdi32.DeleteDC(mem_dc); user32.ReleaseDC(hwnd, hwnd_dc)
    return img

def capture_window_qimage(hwnd: int) -> QtGui.QImage:
    left, top, right, bottom = get_window_rect(hwnd)
    w = max(0, right - left)
    h = max(0, bottom - top)
    if w == 0 or h == 0:
        raise RuntimeError("Window has zero size or is minimized/offscreen.")
    return _capture_via_printwindow_or_bitblt(hwnd, left, top, w, h)

# ======================
#   Overlay Picker
# ======================
class WindowPickerOverlay(QtWidgets.QWidget):
    picked = QtCore.pyqtSignal(int, str)

    def __init__(self, main_hwnd_to_ignore: int = 0):
        super().__init__()
        self._main_hwnd_to_ignore = main_hwnd_to_ignore
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setWindowOpacity(0.15)
        self.setStyleSheet("background: #000;")
        self.setCursor(Qt.CrossCursor)
        vx = user32.GetSystemMetrics(76); vy = user32.GetSystemMetrics(77)
        vw = user32.GetSystemMetrics(78); vh = user32.GetSystemMetrics(79)
        self.setGeometry(vx, vy, vw, vh)
        self.show()
        try:
            HWND_TOPMOST = -1; SWP_NOSIZE = 0x0001; SWP_NOMOVE = 0x0002; SWP_NOACTIVATE = 0x0010
            user32.SetWindowPos(int(self.winId()), HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE|SWP_NOSIZE|SWP_NOACTIVATE)
        except Exception:
            pass

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def _resolve_target_hwnd(self, screen_pos: QtCore.QPoint) -> int:
        self.hide()
        pt = wintypes.POINT(screen_pos.x(), screen_pos.y())
        hwnd = user32.WindowFromPoint(pt)
        hwnd = user32.GetAncestor(hwnd, GA_ROOT)
        my_pid = kernel32.GetCurrentProcessId()
        black_list_classes = {"WorkerW", "Progman", "Shell_TrayWnd"}
        if hwnd:
            if hwnd == int(self.winId()) or hwnd == self._main_hwnd_to_ignore:
                for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),(5,5),(-5,-5)):
                    pt2 = wintypes.POINT(screen_pos.x()+dx, screen_pos.y()+dy)
                    test = user32.WindowFromPoint(pt2)
                    test = user32.GetAncestor(test, GA_ROOT)
                    if test not in (0, int(self.winId()), self._main_hwnd_to_ignore):
                        hwnd = test; break
            cls = get_class_name(hwnd)
            if cls in black_list_classes:
                hwnd = 0
            if hwnd and get_pid(hwnd) == my_pid:
                hwnd = 0
        return hwnd

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == Qt.LeftButton:
            pos = event.globalPos()
            hwnd = self._resolve_target_hwnd(pos)
            self.close()
            if hwnd:
                title = get_window_title(hwnd) or "(untitled)"
                self.picked.emit(hwnd, title)
        else:
            self.close()

# ======================
#   Screen OCR Widget
# ======================
class ScreenOCRWidget(QtWidgets.QWidget):
    frame_captured = QtCore.pyqtSignal(QtGui.QImage)
    meeting_name_updated = QtCore.pyqtSignal(str)

    def __init__(self, log_fn=None):
        super().__init__()
        self._ui_log = log_fn or (lambda s: None)
        self.setWindowTitle("Screen OCR")
        self.resize(900, 620)
        self.capture_btn = QtWidgets.QPushButton("Select Window")
        self.stop_btn = QtWidgets.QPushButton("Stop"); self.stop_btn.setEnabled(False); self.stop_btn.hide()
        self.status = QtWidgets.QLabel("Pick a window to OCR (or leave as default black screen).")
        self.img_label = QtWidgets.QLabel(); self.img_label.setAlignment(Qt.AlignCenter); self.img_label.setMinimumHeight(280)
        self.text_edit = QtWidgets.QPlainTextEdit(); self.text_edit.setPlaceholderText("OCR output will appear here.")

        self.interval_spin = QtWidgets.QSpinBox(); self.interval_spin.setRange(200, 10000); self.interval_spin.setSingleStep(100); self.interval_spin.setValue(1000); self.interval_spin.setSuffix(" ms")
        self.threshold_spin = QtWidgets.QDoubleSpinBox(); self.threshold_spin.setRange(0.0, 1.0); self.threshold_spin.setSingleStep(0.01); self.threshold_spin.setValue(0.01); self.threshold_spin.setDecimals(2)
        self.delta_spin = QtWidgets.QSpinBox(); self.delta_spin.setRange(1, 255); self.delta_spin.setValue(15); self.delta_spin.setSuffix(" Δ")
        self.debug_chk = QtWidgets.QCheckBox("Debug to panel")

        opt_layout = QtWidgets.QHBoxLayout()
        for w in [QtWidgets.QLabel("Interval:"), self.interval_spin, QtWidgets.QLabel("Change ≥"), self.threshold_spin,
                  QtWidgets.QLabel("Pixel Δ:"), self.delta_spin]:
            opt_layout.addWidget(w)
        opt_layout.addStretch(); opt_layout.addWidget(self.debug_chk); opt_layout.addWidget(self.capture_btn); opt_layout.addWidget(self.stop_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(opt_layout)
        layout.addWidget(self.status)
        layout.addWidget(self.img_label, 1)
        layout.addWidget(self.text_edit, 2)

        self.hwnd = None
        self.prev_gray = None
        self.prev_size = None

        self.ocr_lines = []
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self._tick)
        self.capture_btn.clicked.connect(self.start_pick)
        self.stop_btn.clicked.connect(self.stop_monitoring)

        self._log("[init] OCR started; default black screen active.")
        self.stop_btn.setEnabled(True)
        self.timer.start(self.interval_spin.value())
        self._tick(initial=True, force_black=True)

    def _log(self, msg: str):
        # REMOVE: print(msg)
        if self.debug_chk.isChecked():
            self.text_edit.appendPlainText(f"[log] {msg}")
        try:
            self._ui_log(msg)   # this routes to MainWindow._append_log → file
        except Exception:
            pass

    def start_pick(self):
        self.status.setText("Click a window… (ESC/right-click to cancel)")
        self.overlay = WindowPickerOverlay(main_hwnd_to_ignore=int(self.winId()))
        self.overlay.picked.connect(self.on_picked)
        self._log("[pick] Overlay shown. Click target window…")

    def on_picked(self, hwnd: int, title: str):
        try:
            self.hwnd = hwnd
            self.prev_gray = None
            self.prev_size = None
            self.status.setText(f"Selected HWND={hwnd}  Title='{title}'.")
            self._log(f"[pick] Selected HWND={hwnd} Title='{title}'. Monitoring every {self.interval_spin.value()} ms.")
            sanitized = sanitize_meeting_name(title or "meeting")
            self.meeting_name_updated.emit(sanitized)
            self._tick(initial=True)
        except Exception as e:
            self.status.setText(f"Error: {e}")
            self._log(f"[error] on_picked: {e}")

    def stop_monitoring(self):
        self.timer.stop()
        self.stop_btn.setEnabled(False)
        if self.hwnd:
            self.status.setText(f"Stopped monitoring HWND={self.hwnd}.")
            self._log(f"[stop] Stopped monitoring HWND={self.hwnd}.")
        self.hwnd = None
        self.prev_gray = None
        self.prev_size = None

    def _grab_qimage(self) -> QtGui.QImage:
        if self.hwnd:
            return capture_window_qimage(self.hwnd)
        img = QtGui.QImage(1920, 1080, QtGui.QImage.Format.Format_RGB32)
        img.fill(QtGui.QColor(0, 0, 0))
        return img

    def _tick(self, initial=False, force_black=False):
        try:
            img = self._grab_qimage()
        except Exception as e:
            self.status.setText(f"Capture error: {e}")
            self._log(f"[error] capture: {e}")
            return

        self.img_label.setPixmap(QtGui.QPixmap.fromImage(img).scaled(
            self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        QtWidgets.QApplication.processEvents()

        curr_gray = qimage_to_gray_np(img)
        h, w = curr_gray.shape
        sz = (h, w)

        if self.prev_gray is None or self.prev_size != sz or initial:
            self._do_ocr_and_emit(img, reason="first")
            self.prev_gray = curr_gray; self.prev_size = sz
            return

        delta = int(self.delta_spin.value())
        threshold = float(self.threshold_spin.value())
        a = curr_gray.astype(np.int16); b = self.prev_gray.astype(np.int16)
        if a.shape != b.shape:
            self.prev_gray = curr_gray; self.prev_size = sz
            self.status.setText(f"Window size changed to {w}x{h}; baseline reset.")
            self._log("[tick] size changed; baseline reset")
            return

        diff_mask = (np.abs(a - b) > delta)
        change_ratio = float(diff_mask.mean()) if diff_mask.size else 0.0
        self.status.setText(f"Tick: {w}x{h}  change={change_ratio:.2%}  threshold={threshold:.2%}")
        if self.debug_chk.isChecked():
            self._log(f"[tick] size={w}x{h} change={change_ratio:.4f} thr={threshold:.4f} delta={delta}")

        if change_ratio < threshold:
            self.prev_gray = curr_gray; self.prev_size = sz
            return

        self._do_ocr_and_emit(img, reason=f"change {change_ratio:.2%}")
        self.prev_gray = curr_gray; self.prev_size = sz

    def _do_ocr_and_emit(self, img: QtGui.QImage, reason: str):
        if self.hwnd is None:
            self.status.setText("Default black preview (no window selected).")
            return

        try:
            self.frame_captured.emit(img)
        except Exception:
            pass
        try:
            if not USE_TESS:
                self.status.setText("OCR disabled (no Tesseract). Frames still archived.")
                return
            import pytesseract
            from PIL import Image
        except Exception:
            self.status.setText("OCR disabled (pytesseract/PIL not available).")
            return
        try:
            img_rgba = img.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
            width, height = img_rgba.width(), img_rgba.height()
            ptr = img_rgba.bits(); ptr.setsize(img_rgba.sizeInBytes())
            pil = Image.frombuffer("RGBA", (width, height), bytes(ptr), "raw", "RGBA", 0, 1)
            text = pytesseract.image_to_string(pil).strip()
        except Exception as e:
            text = ""
            self._log(f"[error] OCR: {e}")

        ts = datetime.now().strftime("%H:%M:%S")
        if text:
            line = f"[{ts}] {text}"
            self.ocr_lines.append(line)
            self.text_edit.appendPlainText(line)
            self.status.setText(f"OCR appended ({reason}).")
            if self.debug_chk.isChecked():
                self._log(f"[ocr] appended len={len(text)} ({reason})")
        else:
            self.status.setText(f"OCR empty ({reason}).")
            if self.debug_chk.isChecked():
                self._log(f"[ocr] empty ({reason})")

    def get_ocr_text(self) -> str:
        return "\n".join(self.ocr_lines)

    def shutdown(self):
        try: self.timer.stop()
        except Exception: pass

# ======================
#     Audio Widget
# ======================
class AudioTranscriberWidget(QtWidgets.QWidget):
    recording_started = QtCore.pyqtSignal(float)
    recording_stopped = QtCore.pyqtSignal(str, float)
    ui_log = QtCore.pyqtSignal(str)

    def __init__(self, model, log_fn=None, get_ocr_transcript_fn=None):
        super().__init__()
        log.info("Entered AudioTranscriberWidget.__init__"); safe_flush()
        self._ui_log = log_fn or (lambda s: None)
        self.ui_log.connect(self._ui_log)
        self._get_ocr_transcript = get_ocr_transcript_fn or (lambda: "")

        self.debug = DEBUG_MODE
        self.mic_bytes_total = 0
        self.spk_bytes_total = 0
        self.last_meter_log = time.time()
        self.force_transcription_interval = 8  # was 20; faster fallback
        self.mix_mic_gain = 1.0
        self.mix_spk_gain = 1.0

        self.last_transcription_time = time.time()

        self.transcription_lock = Lock()
        self.device_switch_lock = Lock()
        self.stream_lock = Lock()
        self.buffer_lock = Lock()
        self.opening_lock = Lock()

        self.transcription_in_progress = Event()
        self.stop_event = Event()
        self.suspend_transcription = Event()
        self.transcription_scheduled = Event()

        self.vad = webrtcvad.Vad(VAD_MODE)

        # Deques: merged (legacy), plus per-source shadows
        self.buffer = deque(maxlen=CHUNKS_PER_SECOND * BUFFER_SECONDS)         # merged (bytes @ 16k mono)
        self.buffer_mic = deque(maxlen=CHUNKS_PER_SECOND * BUFFER_SECONDS)    # mic-only (bytes @ native SR mono)
        self.buffer_spk = deque(maxlen=CHUNKS_PER_SECOND * BUFFER_SECONDS)    # spk-only (bytes @ native SR mono)

        self.audio_frames = []
        self.was_speech = False

        self.model = model
        self.transcript_lines = []
        self.transcription_hints = deque(maxlen=10)
        self.transcription_counter = 0

        # Audio interface and two (optional) input devices
        self.pa = pyaudio.PyAudio()
        self.stream_mic = None
        self.stream_spk = None
        self.mic_index = None
        self.spk_index = None

        self.channels = 1
        self.sample_rate = 16000             # we write/pipe MIXED @ 16k mono
        self.chunk_ms = CHUNK_DURATION_MS
        self.chunk_16k = int(round(16000 * self.chunk_ms / 1000.0))  # ~30ms frames

        self.audio_thread = None
        self.recording = False
        self.wav_writer = None
        self.audio_start_epoch = None
        self.recording_path = None
        self.samples_written = 0
        self._audio_time_last = 0.0

        self.mic_open = {}
        self.spk_open = {}

        self.setAcceptDrops(True)
        self.init_ui()
        self.refresh_devices()
        self.init_audio_defaults()

        self.force_timer = QtCore.QTimer(self)
        self.force_timer.setInterval(1000)
        self.force_timer.timeout.connect(self.check_force_transcription)
        self.force_timer.start()

        log.info("Audio widget init; dual-source audio thread will start soon.")

    # ----------------- UI -----------------
    def init_ui(self):
        self.status_label = QtWidgets.QLabel("Status: Silent")
        if self.model is None:
            self.status_label.setText("Status: Loading speech model… live audio will buffer until ready.")

        # Two dropdowns
        self.mic_combo = QtWidgets.QComboBox()
        self.spk_combo = QtWidgets.QComboBox()
        self.mic_combo.activated[int].connect(self.on_mic_changed)
        self.spk_combo.activated[int].connect(self.on_spk_changed)

        self.refresh_button = QtWidgets.QPushButton("🔄 Refresh Devices")
        self.refresh_button.clicked.connect(self.refresh_devices)

        self.load_file_button = QtWidgets.QPushButton("📁 Transcribe File…")
        self.load_file_button.clicked.connect(self._switch_to_file_input)

        self.transcript_box = QtWidgets.QPlainTextEdit()
        self.transcript_box.setReadOnly(True)
        self.transcript_box.setWordWrapMode(QtGui.QTextOption.WordWrap)
        self.transcript_box.document().setMaximumBlockCount(15000)

        self.copy_button = QtWidgets.QPushButton("Copy Transcript")
        self.copy_button.setEnabled(False)
        self.copy_button.clicked.connect(self.copy_transcript)
        self.copy_button.setShortcut("Ctrl+C")

        # Keep manual start/stop hidden (auto-managed)
        self.btn_start_rec = QtWidgets.QPushButton("Start Recording"); self.btn_start_rec.hide()
        self.btn_stop_rec  = QtWidgets.QPushButton("Stop Recording");  self.btn_stop_rec.hide()

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.status_label)

        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel("Mic →"))
        row1.addWidget(self.mic_combo, 1)
        row1.addSpacing(12)
        row1.addWidget(QtWidgets.QLabel("Speaker →"))
        row1.addWidget(self.spk_combo, 1)
        row1.addSpacing(12)
        row1.addWidget(self.refresh_button)
        row1.addWidget(self.load_file_button)
        layout.addLayout(row1)

        layout.addWidget(self.transcript_box)
        layout.addWidget(self.copy_button)

        self.mic_combo.setEnabled(True)
        self.spk_combo.setEnabled(True)
        self.copy_button.setEnabled(False)

    def _prefer_index(self, items, *needles):
        """
        Return the first item matching any needle in `label.lower()`.
        Fallback to first list item or None.
        """
        if not items:
            return None
        lname = [d["label"].lower() for d in items]
        for n in needles:
            n = (n or "").lower()
            for i, s in enumerate(lname):
                if n in s:
                    return i
        return 0  # first available

    def _current_combo_data(self, combo):
        idx = combo.currentIndex()
        return combo.itemData(idx) if idx >= 0 else None

    # ----------------- Devices -----------------
    def _enumerate_devices(self):
        mics, spks = [], []
        try:
            count = self.pa.get_device_count()
        except Exception:
            log.exception("Failed to get device count.")
            return mics, spks
        for i in range(count):
            try:
                info = self.pa.get_device_info_by_index(i)
                name = info.get('name', 'Unknown')
                max_input = int(info.get('maxInputChannels', 0))
                if max_input <= 0:
                    continue
                is_loopback = ('loopback' in name.lower()) or bool(info.get('isLoopbackDevice', False))
                pretty = name
                if "loopback" not in name.lower():
                    pretty = f"{name} [{'Loopback' if is_loopback else 'Mic'}]"
                d = {
                    "index": i,
                    "label": pretty,
                    "default_sr": int(info.get('defaultSampleRate', 44100)),
                    "max_in": max_input,
                    "is_loopback": is_loopback
                }

                (spks if is_loopback else mics).append(d)
            except Exception:
                log.exception(f"Error accessing device {i}")
        mics.sort(key=lambda d: d["label"].lower())
        spks.sort(key=lambda d: d["label"].lower())
        return mics, spks

    def refresh_devices(self):
        log.info("Refreshing audio devices..."); safe_flush()
        try:
            with self.opening_lock:
                try: self.pa.terminate()
                except Exception: pass
                self.pa = pyaudio.PyAudio()

            mics, spks = self._enumerate_devices()
            self._mics = mics
            self._spks = spks

            def fill(combo, items):
                combo.blockSignals(True)
                cur = combo.currentText()
                combo.clear()
                combo.addItem("— None —", userData=None)
                for d in items:
                    combo.addItem(d["label"], userData=d)
                if cur and cur in [combo.itemText(i) for i in range(combo.count())]:
                    combo.setCurrentText(cur)
                combo.blockSignals(False)

            fill(self.mic_combo, mics)
            fill(self.spk_combo, spks)
            self.status_label.setText("Status: Devices refreshed.")
            self._log_ui("Audio: device lists refreshed (Mic/Speaker).")
        except Exception:
            log.exception("Device refresh failed.")

    def init_audio_defaults(self):
        """
        Restore old behavior:
          - Mic: prefer Jabra; else any mic
          - Speaker (loopback): prefer Jabra; else "speakers"; else any loopback
        """
        # Mic
        if self.mic_combo.count() > 1:
            mic_items = [self.mic_combo.itemData(i) for i in range(1, self.mic_combo.count())]
            mi = self._prefer_index(mic_items, "jabra")
            self.mic_combo.setCurrentIndex(mi + 1)

        # Speaker (loopback)
        if self.spk_combo.count() > 1:
            spk_items = [self.spk_combo.itemData(i) for i in range(1, self.spk_combo.count())]
            si = self._prefer_index(spk_items, "jabra", "speakers", "remote")
            self.spk_combo.setCurrentIndex(si + 1)

        # Cache indexes
        mic = self._current_combo_data(self.mic_combo)
        spk = self._current_combo_data(self.spk_combo)
        self.mic_index = mic.get("index") if mic else None
        self.spk_index = spk.get("index") if spk else None

        def set_model(self, model):
            self.model = model
            self._log_ui("Speech model ready. Transcription will begin shortly.")
            self._try_schedule_transcription()

        # ----------------- Stream open/close -----------------
        def _try_open_single(self, dev, label):
            """Return dict {stream, sr, ch, fpb} or None if no device."""
            if not dev:
                return None
            info = self.pa.get_device_info_by_index(dev["index"])
            max_in = int(info.get('maxInputChannels', 0))
            if max_in <= 0:
                return None
            default_sr = int(info.get('defaultSampleRate', 44100))
            sr_candidates = []
            for sr in (default_sr, 48000, 44100, 32000, 16000, 8000):
                if sr and sr not in sr_candidates:
                    sr_candidates.append(sr)
            ch_candidates = [min(max_in, 2), 1] if max_in >= 2 else [1]
            last_err = None
            for sr in sr_candidates:
                for ch in ch_candidates:
                    try:
                        fpb = int(sr * self.chunk_ms / 1000.0)
                        s = self.pa.open(
                            format=FORMAT, channels=ch, rate=sr,
                            input=True, frames_per_buffer=fpb,
                            input_device_index=dev["index"]
                        )
                        self._log_ui(f"Audio[{label}]: opened @ {sr} Hz, ch={ch}, fpb={fpb}")
                        return {"stream": s, "sr": sr, "ch": ch, "fpb": fpb}
                    except Exception as e:
                        last_err = e
            self._log_ui(f"Audio[{label}]: failed to open (last error: {last_err})")
            return None

        def start_audio_thread(self):
            if getattr(self, "transcribing_file", False):
                return
            if self.audio_thread is not None and self.audio_thread.is_alive():
                return

            def open_and_run():
                try:
                    with self.opening_lock:
                        mic_dev = self.mic_combo.currentData()
                        spk_dev = self.spk_combo.currentData()
                        mic_state = self._try_open_single(mic_dev, "MIC") if mic_dev else None
                        spk_state = self._try_open_single(spk_dev, "SPK") if spk_dev else None
                        with self.stream_lock:
                            self.stream_mic = mic_state["stream"] if mic_state else None
                            self.stream_spk = spk_state["stream"] if spk_state else None
                            self.mic_open = mic_state or {}
                            self.spk_open = spk_state or {}

                    if not self.stream_mic and not self.stream_spk:
                        self.status_label.setText("Status: No input streams. Pick Mic and/or Speaker.")
                        self.copy_button.setEnabled(bool(self.transcript_lines))
                        return

                    # Begin recording to a 16 kHz mono WAV (merged)
                    self.begin_recording_if_ready()

                    self.stop_event.clear()
                    self.audio_thread = Thread(target=self.audio_loop, name="AudioLoopDual", daemon=True)
                    self.audio_thread.start()
                    self._log_ui("Audio: streams opened and audio thread started (dual-capable).")
                    self.copy_button.setEnabled(True)
                except Exception:
                    log.exception("start_audio_thread: error")

            QtCore.QTimer.singleShot(0, open_and_run)

        def begin_recording_if_ready(self):
            if self.recording:
                return
            _ensure_dir(TEMP_DIR); _ensure_dir(TEMP_FRAMES)
            try:
                ww = wave.open(str(TEMP_WAV), "wb")
                ww.setnchannels(1)
                ww.setsampwidth(SAMPLE_WIDTH)
                ww.setframerate(16000)         # merged file @ 16 kHz mono
                self.wav_writer = ww
                self.recording = True
                self.audio_start_epoch = time.time()
                self.recording_path = str(TEMP_WAV)
                self.samples_written = 0
                self._audio_time_last = 0.0
                self.ui_log.emit(f"Audio: recording started → {TEMP_WAV.name}")
                self.recording_started.emit(self.audio_start_epoch)
            except Exception as e:
                self.wav_writer = None
                self.recording = False
                logging.getLogger("transcriber").exception(f"Auto record start failed: {e}")

        # ----------------- Device change handlers -----------------
        def on_mic_changed(self, _i):
            # Debounce rapid changes
            if not self.device_switch_lock.acquire(blocking=False):
                return
            QtCore.QTimer.singleShot(0, lambda: self._safe_switch(reopen_mic=True, reopen_spk=False))

        def on_spk_changed(self, _i):
            if not self.device_switch_lock.acquire(blocking=False):
                return
            QtCore.QTimer.singleShot(0, lambda: self._safe_switch(reopen_mic=False, reopen_spk=True))

        def _safe_switch(self, reopen_mic=False, reopen_spk=False):
            try:
                self.suspend_transcription.set()
                self.status_label.setText("Status: Switching device(s)…")
                QtCore.QCoreApplication.processEvents()

                # Pause loop cleanly
                self.stop_event.set()
                t = self.audio_thread
                if t and t.is_alive():
                    t.join(timeout=1.0)  # short wait; loop exits fast

                # Close selected streams only
                with self.opening_lock:
                    with self.stream_lock:
                        if reopen_mic and self.stream_mic:
                            try: self.stream_mic.stop_stream()
                            except Exception: pass
                            try: self.stream_mic.close()
                            except Exception: pass
                            self.stream_mic = None
                        if reopen_spk and self.stream_spk:
                            try: self.stream_spk.stop_stream()
                            except Exception: pass
                            try: self.stream_spk.close()
                            except Exception: pass
                            self.stream_spk = None

                # Reopen requested sides
                with self.opening_lock:
                    if reopen_mic:
                        mic_dev = self._current_combo_data(self.mic_combo)
                        mic_state = self._try_open_single(mic_dev, "MIC") if mic_dev else None
                        with self.stream_lock:
                            self.stream_mic = mic_state["stream"] if mic_state else None
                            self.mic_open = mic_state or {}
                    if reopen_spk:
                        spk_dev = self._current_combo_data(self.spk_combo)
                        spk_state = self._try_open_single(spk_dev, "SPK") if spk_dev else None
                        with self.stream_lock:
                            self.stream_spk = spk_state["stream"] if spk_state else None
                            self.spk_open = spk_state or {}

                # If neither reopened successfully, keep running with whatever we have (or idle).
                self.stop_event.clear()
                self.audio_thread = Thread(target=self.audio_loop, name="AudioLoopDual", daemon=True)
                self.audio_thread.start()
                self.status_label.setText("Status: Device(s) switched.")
            except Exception:
                log.exception("Device switch failed")
                self.status_label.setText("Status: Device switch failed (see log).")
            finally:
                self.suspend_transcription.clear()
                try: self.device_switch_lock.release()
                except Exception: pass

                # ----------------- Audio loop (dual source) -----------------
                def audio_loop(self):
                    self._log_ui("Audio: loop entered (dual-source)")
                    VAD_SR = 16000
                    VAD_SAMPLES_30MS = int(VAD_SR * 0.03)

                    def to_mono_i16(raw, ch):
                        if not raw:
                            return None
                        try:
                            x = np.frombuffer(raw, dtype=np.int16)
                            ch = max(1, int(ch or 1))
                            if ch > 1:
                                x = x.reshape(-1, ch).mean(axis=1).astype(np.int16)
                            return x
                        except Exception:
                            return None

                    def i16_to_f32(x):
                        return (x.astype(np.float32) / 32768.0) if x is not None and x.size else None

                    def rms_db(x):
                        if x is None or not x.size:
                            return -120.0
                        r = float(np.sqrt(np.mean(np.square(x), dtype=np.float64)) + 1e-12)
                        return 20.0 * math.log10(r + 1e-20)

                    # local snapshots of streams/params
                    with self.stream_lock:
                        s_m = getattr(self, "stream_mic", None)
                        s_s = getattr(self, "stream_spk", None)
                    m_state = dict(getattr(self, "mic_open", {}) or {})
                    s_state = dict(getattr(self, "spk_open", {}) or {})

                    # frames-per-buffer per device
                    m_fpb = int(m_state.get("fpb", 0) or 0)
                    s_fpb = int(s_state.get("fpb", 0) or 0)

                    while not self.stop_event.is_set():
                        # Pull a block from each *if available*
                        mic_block = None
                        spk_block = None

                        with self.stream_lock:
                            s_m = getattr(self, "stream_mic", None)
                            s_s = getattr(self, "stream_spk", None)

                        try:
                            if s_m and s_m.is_active() and m_fpb > 0:
                                mic_block = s_m.read(m_fpb, exception_on_overflow=False)
                                self.mic_bytes_total += len(mic_block or b"")
                        except Exception:
                            mic_block = None

                        try:
                            if s_s and s_s.is_active() and s_fpb > 0:
                                spk_block = s_s.read(s_fpb, exception_on_overflow=False)
                                self.spk_bytes_total += len(spk_block or b"")
                        except Exception:
                            spk_block = None

                        # Convert to mono i16
                        mic_i16 = to_mono_i16(mic_block, m_state.get("ch", 1)) if mic_block else None
                        spk_i16 = to_mono_i16(spk_block, s_state.get("ch", 1)) if spk_block else None

                        # Resample to 16k float32 for VAD + mix
                        m_sr = int(m_state.get("sr", 16000) or 16000)
                        s_sr = int(s_state.get("sr", 16000) or 16000)
                        mic_f = resample_to_16k(i16_to_f32(mic_i16), m_sr) if mic_i16 is not None else None
                        spk_f = resample_to_16k(i16_to_f32(spk_i16), s_sr) if spk_i16 is not None else None

                        # Log meters occasionally so we can see if mic is alive
                        now = time.time()
                        if now - self.last_meter_log > 5.0:
                            db_m = rms_db(mic_f) if mic_f is not None else -120.0
                            db_s = rms_db(spk_f) if spk_f is not None else -120.0
                            self._log_ui(f"Meter: MIC {db_m:.1f} dBFS, SPK {db_s:.1f} dBFS "
                                         f"(tot mic {self.mic_bytes_total}B, spk {self.spk_bytes_total}B)")
                            self.last_meter_log = now

                        # Per-source VAD (use last 30ms of each if present)
                        def vad_last30(xf):
                            if xf is None or xf.size < VAD_SAMPLES_30MS:
                                return False
                            # convert float back to int16 for webrtcvad
                            y16 = np.clip(np.round(xf[-VAD_SAMPLES_30MS:]*32768.0), -32768, 32767).astype(np.int16)
                            try:
                                return self.vad.is_speech(y16.tobytes(), VAD_SR)
                            except Exception:
                                return False

                        mic_speech = vad_last30(mic_f)
                        spk_speech = vad_last30(spk_f)

                        # Build the mixed float32 16k chunk (align lengths)
                        mix = None
                        if mic_f is not None and spk_f is not None:
                            n = min(mic_f.size, spk_f.size)
                            if n > 0:
                                mix = (self.mix_mic_gain * mic_f[:n]) + (self.mix_spk_gain * spk_f[:n])
                                # prevent clipping
                                mix = np.clip(mix, -1.0, 1.0)
                        elif mic_f is not None:
                            mix = self.mix_mic_gain * mic_f
                        elif spk_f is not None:
                            mix = self.mix_spk_gain * spk_f

                        # Append mixed audio to the transcription buffer in raw int16 (16k)
                        if mix is not None and mix.size:
                            i16 = np.clip(np.round(mix * 32768.0), -32768, 32767).astype(np.int16).tobytes()
                            with self.buffer_lock:
                                self.buffer.append(i16)

                        # UI + scheduling
                        if mic_speech or spk_speech:
                            QtCore.QMetaObject.invokeMethod(
                                self.status_label, "setText", Qt.QueuedConnection,
                                QtCore.Q_ARG(str, "Status: Listening…")
                            )
                            self.was_speech = True
                            # trigger immediately when either talks
                            self._try_schedule_transcription()
                            # Hint for tagging (which source spoke last)
                            if mic_speech:
                                self.transcription_hints.append("[MIC]")
                            if spk_speech:
                                self.transcription_hints.append("[SPK]")
                        else:
                            if self.was_speech:
                                self.was_speech = False
                                QtCore.QMetaObject.invokeMethod(
                                    self.status_label, "setText", Qt.QueuedConnection,
                                    QtCore.Q_ARG(str, "Status: Silent")
                                )
                            # time-based fallback: if buffer has data, flush every few seconds
                            if (now - self.last_transcription_time) > self.force_transcription_interval:
                                with self.buffer_lock:
                                    if len(self.buffer) > 0:
                                        self._try_schedule_transcription()

                        time.sleep(0.002)

                                # ----------------- Transcription -----------------
                                def _try_schedule_transcription(self):
                                    if self.suspend_transcription.is_set():
                                        return
                                    if self.transcription_in_progress.is_set() or self.transcription_scheduled.is_set():
                                        return
                                    self.transcription_scheduled.set()
                                    def run():
                                        try:
                                            self.transcribe_buffer()
                                        finally:
                                            self.transcription_scheduled.clear()
                                    Thread(target=run, name="TranscribeBuffer", daemon=True).start()

                                @contextmanager
                                def transcription_guard(self):
                                    self.transcription_in_progress.set()
                                    try:
                                        yield
                                    finally:
                                        self.transcription_in_progress.clear()

                                def check_force_transcription(self):
                                    if self.stop_event.is_set() or self.suspend_transcription.is_set() or self.transcription_in_progress.is_set():
                                        return
                                    with self.buffer_lock:
                                        buf_has_data = len(self.buffer) > 0
                                    if not buf_has_data:
                                        return
                                    now = time.time()
                                    if (now - self.last_transcription_time) > self.force_transcription_interval:
                                        self._try_schedule_transcription()

                                def transcribe_buffer(self):
                                    with self.transcription_guard():
                                        if self.model is None:
                                            self._log_ui("Model still loading… keeping audio buffered.")
                                            time.sleep(0.25)
                                            return

                                        self.last_transcription_time = time.time()

                                        # Pull snapshots
                                        with self.buffer_lock:
                                            merged_bytes = b"".join(self.buffer); self.buffer.clear()
                                            mic_bytes = b"".join(self.buffer_mic); self.buffer_mic.clear()
                                            spk_bytes = b"".join(self.buffer_spk); self.buffer_spk.clear()

                                        if not (merged_bytes or mic_bytes or spk_bytes):
                                            self._log_ui("Transcribe skip: buffers empty.")
                                            return

                                        # Build per-source 16k float arrays
                                        merged_f32_16k = i16_bytes_to_f32_16k(merged_bytes, 16000) if merged_bytes else np.zeros(0, np.float32)
                                        mic_f32_16k = i16_bytes_to_f32_16k(mic_bytes, int(self.mic_open.get("sr", 16000))) if mic_bytes else np.zeros(0, np.float32)
                                        spk_f32_16k = i16_bytes_to_f32_16k(spk_bytes, int(self.spk_open.get("sr", 16000))) if spk_bytes else np.zeros(0, np.float32)

                                        try:
                                            # Case A: both sources → tag + interleave by timestamps
                                            if mic_f32_16k.size >= 1600 and spk_f32_16k.size >= 1600:
                                                mic_segs, _ = self.model.transcribe(mic_f32_16k, language="en", beam_size=1, vad_filter=False, without_timestamps=False)
                                                spk_segs, _ = self.model.transcribe(spk_f32_16k, language="en", beam_size=1, vad_filter=False, without_timestamps=False)
                                                lines = interleave_tagged_segments(mic_segs, spk_segs)
                                                if not lines and merged_f32_16k.size >= 1600:
                                                    # fallback: merged (untagged)
                                                    m_segs, _ = self.model.transcribe(merged_f32_16k, language="en", beam_size=1, vad_filter=False, without_timestamps=True)
                                                    lines = [seg.text.strip() for seg in m_segs if getattr(seg, "text", "").strip()]
                                            else:
                                                # Case B: only one source available → tag appropriately
                                                if mic_f32_16k.size >= 1600:
                                                    mic_segs, _ = self.model.transcribe(mic_f32_16k, language="en", beam_size=1, vad_filter=False, without_timestamps=True)
                                                    lines = [f"[MIC] {seg.text.strip()}" for seg in mic_segs if getattr(seg, "text", "").strip()]
                                                elif spk_f32_16k.size >= 1600:
                                                    spk_segs, _ = self.model.transcribe(spk_f32_16k, language="en", beam_size=1, vad_filter=False, without_timestamps=True)
                                                    lines = [f"[SPK] {seg.text.strip()}" for seg in spk_segs if getattr(seg, "text", "").strip()]
                                                else:
                                                    # Last resort: merged
                                                    if merged_f32_16k.size < 1600:
                                                        self._log_ui("Transcribe skip: insufficient audio.")
                                                        return
                                                    m_segs, _ = self.model.transcribe(merged_f32_16k, language="en", beam_size=1, vad_filter=False, without_timestamps=True)
                                                    lines = [seg.text.strip() for seg in m_segs if getattr(seg, "text", "").strip()]

                                            if lines:
                                                ts = datetime.now().strftime("%H:%M:%S")

                                                # look back over recent hints to choose a label for untagged lines
                                                label = ""
                                                try:
                                                    for hint in reversed(list(self.transcription_hints)[-5:]):
                                                        if hint in ("[MIC]", "[SPK]"):
                                                            label = hint + " "
                                                            break
                                                except Exception:
                                                    pass

                                                for tline in lines:
                                                    # don't double-tag if the model already produced a [MIC]/[SPK] line
                                                    if tline.startswith("[MIC]") or tline.startswith("[SPK]"):
                                                        prefix = ""
                                                    else:
                                                        prefix = label
                                                    self._write_audio_line(f"[{ts}] {prefix}{tline}")

                                        except Exception:
                                            logging.getLogger("transcriber").exception("Transcription Error")
                                        finally:
                                            self.transcription_counter += 1

                                # ----------------- Misc (unchanged-ish) -----------------
                                def _log_ui(self, msg: str):
                                    try: self._ui_log(msg)
                                    except Exception: pass

                                def _write_audio_line(self, line: str):
                                    self.transcript_lines.append(line)
                                    QtCore.QMetaObject.invokeMethod(
                                        self.transcript_box, "appendPlainText",
                                        Qt.QueuedConnection, QtCore.Q_ARG(str, line)
                                    )

                                def get_audio_time(self) -> float | None:
                                    # we write merged @16k mono; compute time from samples_written
                                    if self.recording:
                                        t = float(self.samples_written) / 16000.0
                                        if t < self._audio_time_last:
                                            t = self._audio_time_last
                                        self._audio_time_last = t
                                        return t
                                    return None

                                def copy_transcript(self):
                                    audio_text = '\n'.join(self.transcript_lines) if self.transcript_lines else ""
                                    try:
                                        ocr_text = self._get_ocr_transcript() or ""
                                    except Exception:
                                        ocr_text = ""
                                    try:
                                        if self.transcript_lines:
                                            start_str = self.transcript_lines[0][1:9]
                                            end_str = self.transcript_lines[-1][1:9]
                                            today = datetime.today()
                                            start_dt = datetime.combine(today, datetime.strptime(start_str, "%H:%M:%S").time())
                                            end_dt = datetime.combine(today, datetime.strptime(end_str, "%H:%M:%S").time())
                                            if end_dt < start_dt:
                                                end_dt += timedelta(days=1)
                                            duration = str(end_dt - start_dt)
                                        else:
                                            start_str = datetime.now().strftime("%H:%M:%S")
                                            duration = "Unknown"
                                    except Exception:
                                        start_str = datetime.now().strftime("%H:%M:%S")
                                        duration = "Unknown"
                                    header = custom_header.replace("{start}", start_str).replace("{duration}", duration)
                                    full = "\n".join([header, "", "=== AUDIO TRANSCRIPT ===", audio_text or "(no audio transcript captured)",
                                                      "", "=== OCR TRANSCRIPT ===", ocr_text or "(no OCR transcript captured)"])
                                    try:
                                        pyperclip.copy(full)
                                        self._log_ui("Combined (audio + OCR) transcripts copied to clipboard.")
                                    except Exception:
                                        log.exception("Failed to copy combined transcripts to clipboard.")

                                def _switch_to_file_input(self):
                                    self.suspend_transcription.set()
                                    try:
                                        self.status_label.setText("Status: Loading file...")
                                        QtCore.QCoreApplication.processEvents()
                                        options = QtWidgets.QFileDialog.Options()
                                        options |= QtWidgets.QFileDialog.DontUseNativeDialog
                                        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                                            self, "Select Audio/Video File", "",
                                            "Media Files (*.mp3 *.wav *.mp4 *.m4a *.flac *.aac *.ogg *.webm *.mkv);;All Files (*)",
                                            options=options
                                        )
                                        if file_path:
                                            self._log_ui(f"Audio: transcribing file '{os.path.basename(file_path)}'.")
                                            self.transcribe_file(file_path)
                                        else:
                                            self.status_label.setText("Status: File load canceled.")
                                            self._log_ui("Audio: file selection canceled.")
                                    except Exception:
                                        log.exception("Switch to file input failed.")
                                        self.status_label.setText("Status: Error switching to file.")
                                    finally:
                                        self.suspend_transcription.clear()

                                def transcribe_file(self, file_path):
                                    """Use faster-whisper on whole files (audio or video)."""
                                    try:
                                        if self.model is None:
                                            self._log_ui("Model not ready yet; cannot transcribe file.")
                                            return
                                        segments, info = self.model.transcribe(
                                            file_path, language="en", beam_size=1, vad_filter=True, without_timestamps=False
                                        )
                                        for seg in segments:
                                            if getattr(seg, "text", ""):
                                                ts = datetime.now().strftime("%H:%M:%S")
                                                self._write_audio_line(f"[{ts}] {seg.text.strip()}")
                                        self.copy_button.setEnabled(bool(self.transcript_lines))
                                    except Exception:
                                        logging.getLogger("transcriber").exception("File Transcription Error")

                                def stop_recording_if_active(self):
                                    if not self.recording:
                                        return
                                    try:
                                        if self.wav_writer:
                                            self.wav_writer.close()
                                    except Exception:
                                        pass
                                    self.wav_writer = None
                                    self.recording = False
                                    self.ui_log.emit(f"Audio: recording stopped ({self.recording_path})")
                                    if self.recording_path and self.audio_start_epoch is not None:
                                        self.recording_stopped.emit(self.recording_path, self.audio_start_epoch)

                                def shutdown(self):
                                    try:
                                        self.suspend_transcription.set()
                                        self.stop_event.set()
                                        try:
                                            if hasattr(self, "force_timer") and self.force_timer:
                                                self.force_timer.stop()
                                        except Exception:
                                            logging.getLogger("transcriber").exception("Audio force_timer stop failed")

                                        # Close streams
                                        try:
                                            with self.opening_lock:
                                                with self.stream_lock:
                                                    for s in (self.stream_mic, self.stream_spk):
                                                        if s:
                                                            try: s.stop_stream()
                                                            except Exception: pass
                                                            try: s.close()
                                                            except Exception: pass
                                                    self.stream_mic = None
                                                    self.stream_spk = None
                                        except Exception:
                                            logging.getLogger("transcriber").exception("Audio stream close failed")

                                        # Briefly wait transcription
                                        try:
                                            if self.transcription_in_progress.is_set():
                                                waited = 0.0
                                                while self.transcription_in_progress.is_set() and waited < 1.0:
                                                    time.sleep(0.1); waited += 0.1
                                        except Exception:
                                            pass

                                        # Join audio thread
                                        try:
                                            t = self.audio_thread
                                            if t and t.is_alive():
                                                t.join(timeout=0.5)
                                                if t.is_alive():
                                                    logging.getLogger("transcriber").warning("Audio thread still alive; continuing shutdown.")
                                        except Exception:
                                            logging.getLogger("transcriber").exception("Audio thread join failed")

                                        # Terminate PyAudio
                                        try:
                                            if hasattr(self, "pa") and self.pa:
                                                self.pa.terminate()
                                        except Exception:
                                            logging.getLogger("transcriber").exception("PyAudio terminate failed")

                                        # Ensure WAV closed
                                        try:
                                            if self.wav_writer:
                                                self.wav_writer.close()
                                        except Exception:
                                            pass
                                        finally:
                                            self.wav_writer = None
                                            self.recording = False
                                    except Exception:
                                        logging.getLogger("transcriber").exception("Audio widget shutdown error (hardened).")

# ======================
#   Main Window + Export & Cleanup
# ======================
class MainWindow(QtWidgets.QMainWindow):
    model_ready = QtCore.pyqtSignal(object)

    class _UILogHandler(logging.Handler):
        def __init__(self, fn):
            super().__init__()
            self._fn = fn
            self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        def emit(self, record):
            if getattr(record, "from_ui", False):
                return
            try:
                self._fn(self.format(record))
            except Exception:
                pass

    def __init__(self, whisper_model):
        super().__init__()
        self._closing = False
        self._export_done = False
        self.setAttribute(QtCore.Qt.WA_QuitOnClose, True)

        # title w/ your name
        self.setWindowTitle("AV + OCR Suite — Caymran Cummings")

        # set icons (both app-wide and this window)
        try:
            icon_path = _frozen_base_dir() / "av-ocr.ico"
            if icon_path.exists():
                qicon = QtGui.QIcon(str(icon_path))
                qapp = QtWidgets.QApplication.instance()
                if qapp is not None:
                    qapp.setWindowIcon(qicon)   # application icon
                self.setWindowIcon(qicon)       # this window's icon
        except Exception:
            logging.getLogger("transcriber").exception("Failed to set window/app icon")

        self.resize(1280, 780)

        self.output_dir = ROOT_ARCHIVES
        self.session_frames = []
        self.session_frame_counter = 0
        self.last_audio_path = None
        self.last_audio_start = None
        self.last_export_path = None

        self.meeting_name = MEETING_NAME
        self._wall_start = None

        self._build_log_dock()

        self.ocr_tab = ScreenOCRWidget(log_fn=self._append_log)
        self.audio_tab = AudioTranscriberWidget(
            whisper_model,
            log_fn=self._append_log,
            get_ocr_transcript_fn=self._get_ocr_transcript_text
        )

        # Loading banner (text updated)
        self.loading_banner = self._build_loading_banner()

        central = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(central); v.setContentsMargins(0,0,0,0)
        v.setSpacing(8)
        v.addWidget(self.loading_banner)
        tabs = QTabWidget()
        tabs.addTab(self.audio_tab, "🎙️  Live Transcription")
        tabs.addTab(self.ocr_tab,  "🪟  Screen OCR")
        v.addWidget(tabs, 1)
        self.setCentralWidget(central)

        # --- Quick actions toolbar (small folder button) ---
        tb = QtWidgets.QToolBar("Quick")
        tb.setIconSize(QtCore.QSize(18, 18))           # small button
        tb.setMovable(False)
        self.addToolBar(QtCore.Qt.TopToolBarArea, tb)

        open_icon = self.style().standardIcon(QtWidgets.QStyle.SP_DirOpenIcon)
        act_open_arch = QtWidgets.QAction(open_icon, "Open Archives", self)
        act_open_arch.setToolTip(f"Open archives folder:\n{ROOT_ARCHIVES}")
        act_open_arch.triggered.connect(self._open_archives_root)
        tb.addAction(act_open_arch)

        self.model_ready.connect(self._on_model_ready)

        ui_handler = MainWindow._UILogHandler(self._append_log)
        ui_handler.setLevel(logging.INFO)
        logging.getLogger("transcriber").addHandler(ui_handler)
        logging.getLogger("transcriber").setLevel(logging.INFO)
        logging.getLogger("transcriber").propagate = True  # make sure exceptions go to file too


        self.ocr_tab.frame_captured.connect(self._on_frame_captured)
        self.ocr_tab.meeting_name_updated.connect(self._on_meeting_name)
        self.audio_tab.recording_started.connect(self._on_recording_started)
        self.audio_tab.recording_stopped.connect(self._on_recording_stopped)

        QtCore.QTimer.singleShot(1000, self.audio_tab.start_audio_thread)

        self._append_log(f"Run folder: {RUN_DIR}")
        self._append_log(f"Temp folder: {TEMP_DIR}")

    @QtCore.pyqtSlot(object)
    def _on_model_ready(self, model):
        try:
            self.audio_tab.set_model(model)
            self.loading_banner.hide()
            self._append_log("Speech model loaded (faster-whisper).")
        except Exception:
            logging.getLogger("transcriber").exception("Error finalizing model load.")

    def _build_loading_banner(self):
        bar = QtWidgets.QFrame()
        bar.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        layout = QtWidgets.QHBoxLayout(bar); layout.setContentsMargins(12, 6, 12, 6)
        icon = QtWidgets.QLabel("⏳")
        lbl  = QtWidgets.QLabel("Loading speech model in the background… Transcription will start once ready.")
        pbar = QtWidgets.QProgressBar(); pbar.setRange(0, 0)
        pbar.setFixedHeight(10)
        layout.addWidget(icon); layout.addWidget(lbl, 1); layout.addWidget(pbar, 0)
        bar.setStyleSheet("QFrame { background:#fff3cd; border:1px solid #ffe58f; border-radius:6px; }")
        return bar

    def _open_archives_root(self):
        path = str(ROOT_ARCHIVES)
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # opens File Explorer
            else:
                QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))
            self._append_log(f"Opened archives folder: {path}")
        except Exception as e:
            self._append_log(f"[error] could not open archives: {e}")


    def _force_export_now(self):
        if self._export_done:
            self._append_log("[export] already completed; skipping forced export.")
            return
        self._append_log("[export] force export invoked.")
        try:
            try: self.ocr_tab.shutdown()
            except Exception: logging.getLogger("transcriber").exception("force: OCR shutdown failed")

            try: self.audio_tab.stop_recording_if_active()
            except Exception: logging.getLogger("transcriber").exception("force: audio stop failed")

            try: self.audio_tab.shutdown()
            except Exception: logging.getLogger("transcriber").exception("force: audio shutdown failed")

            if self.last_audio_path is None and getattr(self.audio_tab, "recording_path", None):
                self.last_audio_path = self.audio_tab.recording_path
            if (not self.last_audio_path) and Path(TEMP_WAV).exists():
                self.last_audio_path = str(TEMP_WAV)

            try:
                self.audio_tab.flush_pending_transcription()
            except Exception:
                logging.getLogger("transcriber").exception("force: flush failed")

            try:
                self._write_final_transcripts()
            except Exception:
                logging.getLogger("transcriber").exception("force: transcript write failed")

            self._export_artifacts()
            self._export_done = True
            self._append_log("[export] Manual export finished.")
        except Exception as e:
            self._append_log(f"[export] manual export failed: {e}")

    def _get_ocr_transcript_text(self) -> str:
        try:
            return self.ocr_tab.get_ocr_text()
        except Exception:
            return ""

    def _build_log_dock(self):
        self.log_dock = QtWidgets.QDockWidget("Log", self)
        self.log_dock.setObjectName("dock_log")
        self.log_dock.setMinimumWidth(360)
        self.log_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        container = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(container); v.setContentsMargins(8,8,8,8)
        row = QtWidgets.QHBoxLayout()
        self.btn_copy_all = QtWidgets.QPushButton("Copy All")
        self.btn_clear_log = QtWidgets.QPushButton("Clear")
        self.btn_copy_all.hide()
        self.btn_clear_log.hide()
        row.addWidget(self.btn_copy_all); row.addStretch(1); row.addWidget(self.btn_clear_log)
        self.log_edit = QtWidgets.QPlainTextEdit(); self.log_edit.setReadOnly(True); self.log_edit.setMaximumBlockCount(50000)
        v.addLayout(row); v.addWidget(self.log_edit)
        self.log_dock.setWidget(container)
        self.addDockWidget(Qt.RightDockWidgetArea, self.log_dock)
        self.btn_copy_all.clicked.connect(self._copy_all_logs)
        self.btn_clear_log.clicked.connect(self.log_edit.clear)

    def _append_log(self, line: str):
        # Always show in the UI
        QtCore.QMetaObject.invokeMethod(
            self.log_edit,
            "appendPlainText",
            Qt.QueuedConnection,
            QtCore.Q_ARG(str, line)
        )
        # NEW: also persist to the file logger without re-injecting into the UI.
        logging.getLogger("transcriber").info(line, extra={"from_ui": True})

    def _copy_all_logs(self):
        txt = self.log_edit.toPlainText()
        QtWidgets.QApplication.clipboard().setText(txt)

    def _on_meeting_name(self, name: str):
        self.meeting_name = sanitize_meeting_name(name or "meeting")
        self._append_log(f"[meeting] name set → {self.meeting_name}")

    def _ensure_temp_frames(self):
        _ensure_dir(TEMP_FRAMES)

    def _on_recording_started(self, audio_start_epoch: float):
        self._ensure_temp_frames()
        self.session_frames.clear()
        self.session_frame_counter = 0
        self.last_audio_start = audio_start_epoch
        self._append_log(f"Audio recording started @ {datetime.fromtimestamp(audio_start_epoch)}")

    def _on_recording_stopped(self, wav_path: str, audio_start_epoch: float):
        self.last_audio_path = wav_path
        self.last_audio_start = audio_start_epoch
        self._append_log(f"Audio recording stopped: {wav_path}")

    @QtCore.pyqtSlot(QtGui.QImage)
    def _on_frame_captured(self, qimg: QtGui.QImage):
        try:
            t = self.audio_tab.get_audio_time()
            if t is None:
                self._append_log("[frame] ignored (no audio timeline yet)")
                return

            self._ensure_temp_frames()
            self.session_frame_counter += 1
            fn = TEMP_FRAMES / f"frame_{self.session_frame_counter:04d}.jpg"
            qimg.save(str(fn), "JPG", quality=90)
            self.session_frames.append((str(fn), float(t)))
            self._append_log(f"[frame] saved {fn.name} @ t={t:.3f}s (audio)")
        except Exception as e:
            self._append_log(f"[error] saving frame: {e}")

    def _write_final_transcripts(self):
        try:
            ocr_text = self._get_ocr_transcript_text().strip()
            if ocr_text:
                ocr_path = ocr_txt_path(self.meeting_name)
                ocr_path.write_text(ocr_text, encoding="utf-8")
                self._append_log(f"OCR transcript → {ocr_path}")
            else:
                self._append_log("No OCR transcript captured; skipping file.")
        except Exception:
            logging.getLogger("transcriber").exception("Failed to write OCR transcript")

        try:
            audio_text = "\n".join(self.audio_tab.transcript_lines).strip()
            if audio_text:
                atp = audio_txt_path(self.meeting_name)
                atp.write_text(audio_text, encoding="utf-8")
                self._append_log(f"Audio transcript → {atp}")
            else:
                self._append_log("No Audio transcript captured; skipping file.")
        except Exception:
            logging.getLogger("transcriber").exception("Failed to write Audio transcript")

    def _export_artifacts(self):
        self._append_log("[export] entry")
        try:
            wav_path = self.last_audio_path
            if not wav_path and Path(TEMP_WAV).exists():
                wav_path = str(TEMP_WAV)

            n_frames = len(self.session_frames)
            self._append_log(f"[export] frames={n_frames} wav={wav_path!s}")

            def wav_ok(p: str | Path, min_bytes: int = 1024) -> bool:
                try:
                    pp = Path(p) if p else None
                    ok = bool(pp and pp.exists() and pp.stat().st_size >= min_bytes)
                    if not ok:
                        self._append_log(f"[export] WAV missing/too small: {p}")
                    return ok
                except Exception as e:
                    self._append_log(f"[export] WAV check error: {e}")
                    return False

            def has_usable_frames() -> bool:
                if n_frames == 0:
                    return False
                if n_frames == 1:
                    return float(self.session_frames[0][1]) > 0.2
                return True

            have_wav = wav_ok(wav_path)
            usable_frames = has_usable_frames()

            if not usable_frames and not have_wav:
                self._append_log("[export] Nothing to export (no frames and no wav).")
                return

       

            if usable_frames:
                out_mp4 = video_mp4_path(self.meeting_name)
                try:
                    export_archive_build_video(self.session_frames, wav_path if have_wav else None, crf=28, output_path=out_mp4)
                    self.last_export_path = str(out_mp4)
                    self._append_log(f"Archive export complete: {out_mp4}")
                    if have_wav:
                        try: Path(wav_path).unlink(missing_ok=True)
                        except Exception: logging.getLogger("transcriber").exception("Could not delete temp WAV after MP4 export.")
                    return
                except Exception as e:
                    self._append_log(f"[export] MP4 export failed, falling back to MP3: {e}")

            if have_wav:
                out_mp3 = audio_mp3_path(self.meeting_name)
                cmd = ["ffmpeg", "-y", "-i", str(wav_path), "-c:a", "libmp3lame", "-b:a", "192k", str(out_mp3)]
                self._append_log(f"[export] Running ffmpeg (audio-only): {' '.join(cmd)}")
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if proc.stdout: self._append_log(f"[ffmpeg] stdout:\n{proc.stdout}")
                if proc.stderr: self._append_log(f"[ffmpeg] stderr:\n{proc.stderr}")
                if proc.returncode != 0 or not out_mp3.exists():
                    raise RuntimeError(f"ffmpeg mp3 export failed with code {proc.returncode}")
                try: Path(wav_path).unlink(missing_ok=True)
                except Exception: logging.getLogger("transcriber").exception("Could not delete source WAV after MP3 export.")
                self.last_export_path = str(out_mp3)
                self._append_log(f"Audio export complete: {out_mp3}")
            else:
                self._append_log("[export] Skipped MP3 fallback (no WAV found).")

        except Exception as e:
            self._append_log(f"[export] {e}")
            try:
                QMessageBox.warning(self, "Archive Export", f"Could not export archive:\n{e}\n(Temp left in {TEMP_DIR})")
            except Exception:
                pass

    def _cleanup_temps(self):
        def _onerror(func, path, exc_info):
            try:
                os.chmod(path, 0o666)
            except Exception:
                pass
            try:
                if Path(path).is_dir():
                    os.rmdir(path)
                else:
                    os.remove(path)
            except Exception:
                pass

        time.sleep(0.25)

        for attempt in range(6):
            try:
                if TEMP_DIR.exists():
                    shutil.rmtree(TEMP_DIR, onerror=_onerror)
                self._append_log(f"Cleaned temp folder: {TEMP_DIR}")
                break
            except PermissionError:
                time.sleep(0.25)
            except Exception as e:
                if attempt == 5:
                    logging.getLogger("transcriber").exception(f"Failed to clean temp folder after retries: {TEMP_DIR}")
                    self._append_log(f"[cleanup] Could not fully remove {TEMP_DIR} (see log).")
                else:
                    time.sleep(0.25)

    def closeEvent(self, event: QtGui.QCloseEvent):
        if self._closing:
            event.accept(); return
        self._closing = True
        self._append_log("Close event triggered.")

        try:
            # 1) Stop ongoing capture ASAP so WAV is final
            self._append_log("[shutdown] stopping OCR…")
            try: self.ocr_tab.shutdown()
            except Exception: logging.getLogger("transcriber").exception("OCR shutdown failed")

            self._append_log("[shutdown] stopping audio recording (if active)…")
            try: self.audio_tab.stop_recording_if_active()
            except Exception: logging.getLogger("transcriber").exception("Audio stop failed")

            # 2) FLUSH + EXPORT *BEFORE* tearing down PyAudio
            self._append_log("[shutdown] flushing any pending transcription…")
            try:
                self.audio_tab.flush_pending_transcription()
            except Exception:
                logging.getLogger("transcriber").exception("Flush pending transcription failed")

            self._append_log("[shutdown] writing final transcripts…")
            try:
                self._write_final_transcripts()
            except Exception:
                logging.getLogger("transcriber").exception("Final transcript write failed")

            self._append_log("[shutdown] starting export…")
            try:
                self._export_artifacts()
                self._export_done = True
            except Exception:
                logging.getLogger("transcriber").exception("Export failed in closeEvent (early).")

            # 3) Now tear down audio (can be slow / flaky on VPS devices)
            self._append_log("[shutdown] shutting down audio subsystem…")
            try: self.audio_tab.shutdown()
            except Exception:
                logging.getLogger("transcriber").exception("Audio shutdown failed")

        finally:
            self._append_log("[shutdown] cleaning temp folder…")
            try:
                self._cleanup_temps()
            except Exception:
                logging.getLogger("transcriber").exception("Temp cleanup failed in closeEvent.")

            event.accept()
            QtCore.QTimer.singleShot(0, QtWidgets.QApplication.instance().quit)

# ======================
#         Main
# ======================
import atexit
def _export_guard():
    try:
        if Path(TEMP_WAV).exists():
            run_dir = RUN_DIR
            any_final = any(run_dir.glob(f"{STAMP}_video_*.mp4")) or any(run_dir.glob(f"{STAMP}_audio_*.mp3"))
            if not any_final:
                out_mp3 = run_dir / f"{STAMP}_audio_meeting.mp3"
                cmd = ["ffmpeg", "-y", "-i", str(TEMP_WAV), "-c:a", "libmp3lame", "-b:a", "192k", str(out_mp3)]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

atexit.register(_export_guard)

if __name__ == "__main__":

    # Hide pop-up consoles from subprocesses in frozen exe
    if getattr(sys, 'frozen', False):
        _original_popen = subprocess.Popen
        def silent_popen(*args, **kwargs):
            if 'startupinfo' not in kwargs:
                si = subprocess.STARTUPINFO()
                si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                kwargs['startupinfo'] = si
            return _original_popen(*args, **kwargs)
        subprocess.Popen = silent_popen
        try:
            log_file = ROOT_LOGS / f"stdout_{STAMP}.log"
            sys.stdout = open(log_file, "w", encoding="utf-8")
            logger = logging.getLogger("transcriber")
            for h in logger.handlers:
                if isinstance(h, logging.StreamHandler):
                    # make sure it writes to the new stdout file
                    try:
                        h.setStream(sys.stdout)
                    except Exception:
                        pass
            sys.stderr = sys.stdout
        except Exception:
            pass

    # DPI awareness
    if sys.platform.startswith("win"):
        try:
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        except Exception:
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                pass

    try:
        log.info("Starting AV + OCR application…"); safe_flush()

        if sys.platform.startswith("win"):
            try:
                myappid = u'caymran.av.ocr.suite.2.3'
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            except Exception:
                pass

        app = QtWidgets.QApplication(sys.argv)

        def log_ffmpeg_info_once():
            import shutil, subprocess
            p = shutil.which("ffmpeg")
            if not p:
                log.info("ffmpeg not found on PATH")
                return
            try:
                out = subprocess.run([p, "-version"], capture_output=True, text=True, timeout=3)
                first = (out.stdout or out.stderr or "").splitlines()[:2]
                log.info(f"ffmpeg: {p}")
                for line in first:
                    log.info(f"ffmpeg: {line}")
            except Exception as e:
                log.info(f"ffmpeg present but -version failed: {e}")

        # call it:
        _enable_tesseract_if_present()
        _ensure_ffmpeg_on_path()
        log_ffmpeg_info_once()

        # set default application icon early
        try:
            icon_path = _frozen_base_dir() / "av-ocr.ico"
            if icon_path.exists():
                app.setWindowIcon(QtGui.QIcon(str(icon_path)))
        except Exception:
            pass

        # Create window WITHOUT a model so the UI appears instantly
        window = MainWindow(whisper_model=None)  # param name legacy
        app.aboutToQuit.connect(window._force_export_now)
        window.show(); window.raise_(); window.activateWindow()
        log.info("Window shown; loading speech model in the background…"); safe_flush()

        # Background model loader thread (CTRANSLATE2 / faster-whisper)
        from faster_whisper import WhisperModel

        def _load_model_bg():
            try:
                # Prefer explicit override (folder or model name)
                override = os.getenv("AVOS_WHISPER_MODEL", "").strip()

                # Common local locations (packaged or user cache)
                candidates = []
                if override:
                    candidates.append(Path(override))
                candidates += [
                    _frozen_base_dir() / "models" / "base.en",               # bundled with EXE
                    Path.home() / "AppData/Local/AV_OCR_Suite/models/base.en" # per-user cache
                ]

                local_dir = next((p for p in candidates if p and p.exists()), None)

                if local_dir:
                    mdl = WhisperModel(
                        str(local_dir),
                        device="cpu",
                        compute_type=os.getenv("AVOS_COMPUTE", "int8"),
                        local_files_only=True,          # force offline when we have files
                        download_root=str(APP_CACHE),
                        cpu_threads=max(1, os.cpu_count() or 1),
                    )
                else:
                    mdl = WhisperModel(
                        os.getenv("AVOS_WHISPER_MODEL", "base.en"),
                        device="cpu",
                        compute_type=os.getenv("AVOS_COMPUTE", "int8"),
                        local_files_only=False,         # allow online fetch if available
                        download_root=str(APP_CACHE),
                        cpu_threads=max(1, os.cpu_count() or 1),
                    )

                window.model_ready.emit(mdl)

            except Exception:
                logging.getLogger("transcriber").exception("Speech model load failed (faster-whisper)")

        Thread(target=_load_model_bg, name="FWModelLoader", daemon=True).start()

        log.info(f"Run output dir: {RUN_DIR}")
        log.info(f"Temp dir: {TEMP_DIR}")

        sys.exit(app.exec_())

    except Exception:
        logging.getLogger("transcriber").exception("Application Startup Error")
        try:
            print("An error occurred. See logs folder.")
        except Exception:
            pass

        sys.exit(1)


