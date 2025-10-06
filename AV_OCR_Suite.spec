# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_dynamic_libs
from pathlib import Path
import importlib
import os

block_cipher = None

# ---------- Core binaries ----------
binaries = []
binaries += collect_dynamic_libs('pyaudiowpatch')

# Numpy (optional)
try:
    binaries += collect_dynamic_libs('numpy')
except Exception:
    pass

# ctranslate2 native bits (covers .pyd/.dll and .libs layouts)
try:
    ct = importlib.import_module("ctranslate2")
    ct_dir = Path(ct.__file__).parent
    for pattern in ("*.pyd", "*.dll", "dlls/*.dll", "lib/*.dll", "ctranslate2.libs/*.dll"):
        for f in ct_dir.glob(pattern):
            binaries.append((str(f), "."))
except Exception:
    pass

# ---------- App data ----------

# make sure the ico is included as data so Qt can load it at runtime too
datas = [('av-ocr.ico', '.')]

# put README in the dist root
datas += [('README.txt', '.')]   

datas += [
    ('models/base.en/*', 'models/base.en'),
]

# Only include custom_prompt.txt if it’s present in the repo
if Path("custom_prompt.txt").exists():
    datas.append(("custom_prompt.txt", ".", "."))  # (src, destdir, type)

# ---------- Portable Tesseract (optional) ----------
# Point this to your portable Tesseract root folder.
TESS_ROOT = r"C:\AIWorkspaces\portable-tesseract"
if os.path.exists(TESS_ROOT):
    # EXE + DLLs => <app>\tesseract\...
    # Grab tesseract.exe plus any DLLs beside it (robust across builds)
    for pat in ("tesseract.exe", "*.dll"):
        for f in Path(TESS_ROOT).glob(pat):
            if f.is_file():
                binaries.append((str(f), 'tesseract'))
    # Minimal language data
    for lang in ("eng.traineddata", "osd.traineddata"):
        lang_p = Path(TESS_ROOT) / "tessdata" / lang
        if lang_p.exists():
            datas.append((str(lang_p), 'tesseract/tessdata'))

# ---------- Portable FFmpeg (optional) ----------
# Point this to your portable FFmpeg root (static or shared build).
FFMPEG_ROOT = r"C:\AIWorkspaces\portable-ffmpeg"
if os.path.exists(FFMPEG_ROOT):
    ff_root = Path(FFMPEG_ROOT)
    # prefer <root>\bin if present
    bin_dir = ff_root / "bin" if (ff_root / "bin").exists() else ff_root
    # executables
    for exe in ("ffmpeg.exe", "ffprobe.exe", "ffplay.exe"):
        p = bin_dir / exe
        if p.exists():
            binaries.append((str(p), 'ffmpeg'))
    # ship needed DLLs if present (shared builds)
    for dll in bin_dir.glob("*.dll"):
        binaries.append((str(dll), 'ffmpeg'))

# ---------- Hidden imports ----------
hiddenimports = [
    # OCR stack
    'pytesseract',
    'PIL',
    'PIL.Image',
    'PIL._imaging',
    'PIL.PngImagePlugin',
    'PIL.JpegImagePlugin',
    'PIL.BmpImagePlugin',
    # (uncommon, but add if logs later mention tiff) 'PIL.TiffImagePlugin',

    # your existing speech/runtime deps
    'faster_whisper',
    'ctranslate2',
    'huggingface_hub',
    'tokenizers',
    'filelock',
    'packaging',
    'requests',
    'tqdm',
    'asyncio',
    'asyncio.windows_events',
    'asyncio.windows_utils',
]

# ---------- Runtime hooks ----------
# 1) Win asyncio policy (fixes ctranslate2 import under PyInstaller)
# 2) Add packaged ffmpeg/tesseract to PATH and set app-local caches
runtime_hooks = [
    'rthook_asyncio_winpolicy.py',
    'rthook_portables_on_path.py',
]

a = Analysis(
    ['av_ocr_suite.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['.'],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
    excludes=[
        'torch', 'torch.*',
        'torchaudio', 'torchaudio.*',
        'whisper', 'whisper.*',
        'librosa', 'numba', 'numba.*', 'llvmlite', 'llvmlite.*',
        'scipy', 'numpy.f2py',
        'matplotlib', 'IPython',
        'PyQt5.QtQml', 'PyQt5.QtQuick', 'PyQt5.QtQuick3D', 'PyQt5.QtWebEngine',
        'PyQt5.QtMultimedia', 'PyQt5.QtMultimediaWidgets', 'PyQt5.QtNetwork',
        'PyQt5.QtBluetooth', 'PyQt5.QtSerialPort', 'PyQt5.QtPositioning',
        'PyQt5.QtSensors', 'PyQt5.QtNfc', 'PyQt5.QtHelp', 'PyQt5.QtDesigner',
        'PyQt5.QtWebSockets', 'PyQt5.QtWebChannel', 'PyQt5.QtTest',
        'PyQt5.Qt3DAnimation', 'PyQt5.Qt3DCore', 'PyQt5.Qt3DLogic', 'PyQt5.Qt3DRender',
        'PyQt5.QtXml', 'PyQt5.QtXmlPatterns', 'PyQt5.QtSvg', 'PyQt5.QtDBus',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],  # no splash
    exclude_binaries=True,
    name='AV_OCR_Suite',
	icon="av-ocr.ico",              # <-- this embeds the icon resource in the EXE
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AV_OCR_Suite',
)
