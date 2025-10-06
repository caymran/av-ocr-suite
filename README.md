# 🧠 AV + OCR Suite
**Developer:** [Caymran Cummings](https://github.com/caymran)  
**License:** GNU General Public License v3.0  

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)  
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)  
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)  
![Faster Whisper](https://img.shields.io/badge/Speech-FasterWhisper-orange.svg)  

---

## 🎯 Overview
**AV + OCR Suite** is a standalone transcription and visual capture tool that:
- Records system or microphone audio in real-time (loopback or mic input).  
- Captures screenshots of an active window periodically.  
- Runs **speech-to-text** transcription using Faster-Whisper.  
- Performs **on-screen text recognition** using Tesseract OCR.  
- Automatically packages session archives containing audio, video, frames, transcripts, and logs.  

---

## ⚙️ Features
| Category | Description |
|-----------|--------------|
| **Audio Capture** | Live microphone or loopback recording via PyAudio. |
| **Speech Recognition** | Fast Whisper (base.en) model for offline English transcription. |
| **OCR** | Tesseract integration for extracting text from periodic screenshots. |
| **FFmpeg Export** | Merges frames and audio into a synchronized MP4 on exit. |
| **Logging** | Unified on-screen and file logging under `logs/`. |
| **Archives** | Each run generates a self-contained folder under `archives/`. |
| **Installer** | Packaged via Inno Setup for per-user install (no elevation required). |
| **Portable** | Bundled with pre-packaged `ffmpeg` and `tesseract` binaries. |

---

## 🖥️ Installation
1. **Download the installer**  
   From the [Releases](../../releases) page:
   ```
   AV_OCR_Suite_2.3.0_UserSetup.exe
   ```
   > Installs to `%LOCALAPPDATA%\AV_OCR_Suite`

2. **First Run**  
   The app will cache the Whisper model, initialize ffmpeg and tesseract, and create the archive structure.

---

## 🚀 Usage Guide
### 1️⃣ Start the Application
Launch from the Start Menu or desktop shortcut.  

### 2️⃣ Select Audio Source
Choose **Loopback** to record system sound or **Microphone** to capture your voice.

### 3️⃣ Pick a Window
Click **🎯 Select Target Window** and choose any active window.  

### 4️⃣ Transcription
Speech and OCR results appear as logs while frames and audio are stored in `archives/`.

### 5️⃣ Export
Closing the app automatically merges audio + frames into MP4 and saves the session archive.

### 6️⃣ View Archives
Click **📂 Open Archives Folder** to open your saved sessions in Explorer.

---

## 📄 Directory Structure
```
AV_OCR_Suite/
├── AV_OCR_Suite.exe
├── README.txt
├── ffmpeg/
├── tesseract/
├── models/
│   └── base.en/
├── archives/
│   ├── 20250918_113245/
│   │   ├── audio_20250918_113245.wav
│   │   ├── frame_0001.jpg
│   │   ├── transcript.txt
│   │   ├── output.mp4
│   │   └── log.txt
└── logs/
    ├── app_YYYYMMDD_HHMMSS.log
    └── stdout_YYYYMMDD_HHMMSS.log
```

---

## 🔒 License
This project is licensed under **GPL v3.0**  
See [LICENSE](LICENSE) for full text.

---

## 👏 Credits
- **Caymran Cummings** — Author and maintainer  
- **Gyan.dev** — FFmpeg builds  
- **Google Tesseract OCR** — OCR engine  
- **Guillaume Klein** — Faster-Whisper implementation

---

## 💡 Future Ideas
- Dual mic + loopback capture  
- GPU inference mode  
- Subtitle export  
- Meeting summarization  

---

## 📬 Contact
📧 [Caymran Cummings](mailto:caymran@users.noreply.github.com)  
🌐 [GitHub](https://github.com/caymran)
