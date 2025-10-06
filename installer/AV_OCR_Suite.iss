; ===== AV_OCR_Suite - Per-User Installer (no admin) =====
; Installs under %LOCALAPPDATA% and never asks for elevation.
; NOTE: Save this file as UTF-8 WITH BOM to avoid garbled characters.

; ---- Resolve repo root & dist path robustly (works in GitHub Actions & locally) ----
#define RepoRoot GetEnv("GITHUB_WORKSPACE")
#if RepoRoot == ""
  #define RepoRoot GetEnv("CD")
#endif
#if RepoRoot == ""
  #define RepoRoot "."
#endif

#define DistDir   AddBackslash(RepoRoot) + "dist\\AV_OCR_Suite"

#ifndef SourceDir
  #define SourceDir DistDir
#endif

#define AppName        "AV + OCR Suite"
#define AppVersion     "2.3.0"
#define AppPublisher   "Caymran Cummings"
#define AppURL         "https://github.com/caymran/av-ocr-suite"
#define AppId          "{{A6C7E7A5-8B5E-42E6-9D6E-7B5B9F3B1F0E}}"

[Setup]
AppId={#AppId}
AppName={#AppName}
AppVersion={#AppVersion}

; Show your name in the wizard header
AppVerName={#AppName} - by {#AppPublisher} (v{#AppVersion})

; Publisher / metadata
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
VersionInfoCompany={#AppPublisher}
VersionInfoDescription=Setup for {#AppName} by {#AppPublisher}
VersionInfoVersion={#AppVersion}.0

UninstallDisplayName={#AppName} - by {#AppPublisher}
UninstallDisplayIcon={app}\AV_OCR_Suite.exe

; Per-user install
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline
UsedUserAreasWarning=no

; Install under %LOCALAPPDATA%
DefaultDirName={localappdata}\AV_OCR_Suite
DisableDirPage=no

; Start Menu group
DefaultGroupName=AV + OCR Suite
DisableProgramGroupPage=yes

ArchitecturesInstallIn64BitMode=x64

; Cosmetic / behavior
Compression=lzma2
SolidCompression=yes
SetupLogging=yes
OutputBaseFilename=AV_OCR_Suite_{#AppVersion}_UserSetup
WizardStyle=modern
DisableWelcomePage=no
DisableReadyMemo=no

; Use the ICO from the REPO ROOT (safer than from dist)
SetupIconFile={#RepoRoot}\av-ocr.ico

; Optional: prevent multiple instances during install
CloseApplications=yes
CloseApplicationsFilter=*.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
; Copy the PyInstaller output to {app}
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

; Ship the README from the REPO ROOT (not dist)
Source: "{#RepoRoot}\README.txt"; DestDir: "{app}"; Flags: ignoreversion

; include model only if present
Source: "{#SourceDir}\models\base.en\*"; DestDir: "{app}\models\base.en"; Flags: recursesubdirs createallsubdirs ignoreversion; Check: DirExists(ExpandConstant('{#SourceDir}\models\base.en'))


; ffmpeg (optional)
Source: "{#SourceDir}\ffmpeg\*"; DestDir: "{app}\ffmpeg"; Flags: recursesubdirs createallsubdirs ignoreversion; Check: DirExists(ExpandConstant('{#SourceDir}\ffmpeg'))

; Tesseract + tessdata (optional)
Source: "{#SourceDir}\tesseract\*"; DestDir: "{app}\tesseract"; Flags: recursesubdirs createallsubdirs ignoreversion; Check: DirExists(ExpandConstant('{#SourceDir}\tesseract'))

[Icons]
; Start Menu shortcut
Name: "{userprograms}\AV + OCR Suite\AV + OCR Suite"; Filename: "{app}\AV_OCR_Suite.exe"; WorkingDir: "{app}"; IconFilename: "{app}\av-ocr.ico"; IconIndex: 0
; Optional Desktop shortcut (checked by default via Tasks)
Name: "{userdesktop}\AV + OCR Suite"; Filename: "{app}\AV_OCR_Suite.exe"; Tasks: desktopicon; WorkingDir: "{app}"; IconFilename: "{app}\av-ocr.ico"; IconIndex: 0

[Run]
; Launch app after install
Filename: "{app}\AV_OCR_Suite.exe"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
; Show README (checked by default because we didn’t mark it unchecked)
Filename: "{app}\README.txt"; Description: "Show &Readme"; Flags: postinstall shellexec skipifsilent nowait

[UninstallDelete]
Type: filesandordirs; Name: "{localappdata}\AV_OCR_Suite\cache"
