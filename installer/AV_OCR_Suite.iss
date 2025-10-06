; ===== AV_OCR_Suite — Per-User Installer (no admin) =====
; Installs under %LOCALAPPDATA% and never asks for elevation.
; Save this file as UTF-8 WITH BOM to avoid garbled characters (—, ©).

; Directory of the repository root (script is inside /installer)
#define ScriptDir    GetEnv("CD")                     ; current working dir when CI calls ISCC (repo root)
#define DistDir      AddBackslash(ScriptDir) + "dist\AV_OCR_Suite"

#ifndef SourceDir
  #define SourceDir DistDir
#endif

#define AppName        "AV + OCR Suite"
#define AppVersion     "2.3.0"
#define AppPublisher   "Caymran Cummings"
#define AppURL         "https://github.com/caymran/av-ocr-suite"
; Use a stable GUID for upgrades (generate once and keep it)
#define AppId          "{{A6C7E7A5-8B5E-42E6-9D6E-7B5B9F3B1F0E}}"

; Where your built app lives (PyInstaller output)

[Setup]
AppId={#AppId}
AppName={#AppName}
AppVersion={#AppVersion}

; ***** Put your name in the installer window title *****
; (What the wizard shows in its title bar / header)
AppVerName={#AppName} — by {#AppPublisher} (v{#AppVersion})

; ***** Publisher / metadata (Programs & Features, file properties) *****
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
VersionInfoCompany={#AppPublisher}
VersionInfoDescription=Setup for {#AppName} by {#AppPublisher}
; Optional: make file version 4-part
VersionInfoVersion={#AppVersion}.0

; Show your name in the uninstall entry too
UninstallDisplayName={#AppName} — by {#AppPublisher}
UninstallDisplayIcon={app}\AV_OCR_Suite.exe

; ---- Per-user install: no elevation ----
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=commandline
UsedUserAreasWarning=no

; Install under %LOCALAPPDATA%
DefaultDirName={localappdata}\AV_OCR_Suite
DisableDirPage=no

; Put shortcuts in the user's Start Menu group
DefaultGroupName=AV + OCR Suite
DisableProgramGroupPage=yes

; 64-bit binaries are fine installed to user space
ArchitecturesInstallIn64BitMode=x64

; Cosmetic / behavior
Compression=lzma2
SolidCompression=yes
SetupLogging=yes
OutputBaseFilename=AV_OCR_Suite_{#AppVersion}_UserSetup
WizardStyle=modern
DisableWelcomePage=no
DisableReadyMemo=no
; Use a real .ico (multi-size: 16/32/48/256). Avoid “icon file too large” errors.
SetupIconFile={#SourceDir}\av-ocr.ico

; Optional: prevent multiple instances during install
CloseApplications=yes
CloseApplicationsFilter=*.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
; Copy everything from the PyInstaller folder to {app}
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
; Ship the readme with your app
Source: "{#SourceDir}\README.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\models\base.en\*"; DestDir: "{app}\models\base.en"; Flags: recursesubdirs createallsubdirs ignoreversion

; (Optional) Pre-create the user model cache folder if you ship any files there
; Check uses ExpandConstant directly—no custom Code section needed.
; Source: "{#SourceDir}\models_cache\*"; \
;   DestDir: "{localappdata}\AV_OCR_Suite\models_cache"; \
;   Flags: recursesubdirs createallsubdirs ignoreversion; \
;   Check: DirExists(ExpandConstant('{#SourceDir}\models_cache'))

[Icons]
; Start Menu shortcut
Name: "{userprograms}\AV + OCR Suite\AV + OCR Suite"; Filename: "{app}\AV_OCR_Suite.exe"; WorkingDir: "{app}"; IconFilename: "{app}\av-ocr.ico"; IconIndex: 0
; Optional Desktop shortcut
Name: "{userdesktop}\AV + OCR Suite"; Filename: "{app}\AV_OCR_Suite.exe"; Tasks: desktopicon; WorkingDir: "{app}"; IconFilename: "{app}\av-ocr.ico"; IconIndex: 0


[Run]
; Offer to launch after install finishes
Filename: "{app}\AV_OCR_Suite.exe"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
; Checkbox appears on final page; checked by default since we did NOT add "unchecked"
; shellexec opens it with the user's default editor
Filename: "{app}\README.txt"; Description: "Show &Readme"; Flags: postinstall shellexec skipifsilent nowait

[UninstallDelete]
; Clean up per-user caches you own (safe to keep if you prefer)
Type: filesandordirs; Name: "{localappdata}\AV_OCR_Suite\cache"
