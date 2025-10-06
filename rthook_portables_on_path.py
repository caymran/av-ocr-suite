# rthook_portables_on_path.py
import os, sys
from pathlib import Path

# Frozen app root
if getattr(sys, 'frozen', False):
    app_root = Path(sys._MEIPASS).resolve()
else:
    app_root = Path(__file__).resolve().parent

# Prepend packaged folders to PATH so shutil.which() finds them
portable_dirs = []
ffmpeg_dir = app_root / "ffmpeg"
tess_dir   = app_root / "tesseract"
if ffmpeg_dir.exists():
    portable_dirs.append(str(ffmpeg_dir))
if tess_dir.exists():
    portable_dirs.append(str(tess_dir))

if portable_dirs:
    os.environ["PATH"] = os.pathsep.join(portable_dirs + [os.environ.get("PATH", "")])

# App-local caches (keeps EXE tiny; models go to user cache)
app_name = "AV_OCR_Suite"
base_cache = Path(os.environ.get("LOCALAPPDATA", Path.home())) / app_name / "cache"
os.environ.setdefault("HF_HOME", str(base_cache / "hf"))
os.environ.setdefault("CT2_CACHE_DIR", str(base_cache / "ct2"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("CT2_USE_CPU", "1")
