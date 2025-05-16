# -*- mode: python ; coding: utf-8 -*-
"""ve.spec – PyInstaller recipe for *VirtualElevationRecipes*

* Builds an **onedir** bundle on Windows & Linux.
* On **macOS** it additionally wraps the executable in a native `.app` bundle
  via the `BUNDLE` command so the workflow can create a DMG.
* Assumes the repo root contains `main.py`, `VE_icon.(png|icns|ico)`, and any
  optional resource folders (`ui/`, `models/`, `utils/`, `config/`).

Run with:

    pyinstaller ve.spec --clean --noconfirm
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

APP_NAME = "VirtualElevationRecipes"
block_cipher = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPEC_DIR = Path.cwd()  # Use current working directory directly
ENTRY_SCRIPT = SPEC_DIR / "main.py"
if not ENTRY_SCRIPT.exists():
    raise SystemExit(f"❌ main.py not found at {ENTRY_SCRIPT}")

# ---------------------------------------------------------------------------
# Datas – copy only if they exist
# ---------------------------------------------------------------------------
DATA_FOLDERS = ["ui", "models", "utils", "config"]

datas = [
    (str(SPEC_DIR / f), f) for f in DATA_FOLDERS if (SPEC_DIR / f).exists()
]

for icon in ("VE_icon.png", "VE_icon.ico", "VE_icon.icns"):
    p = SPEC_DIR / icon
    if p.exists():
        datas.append((str(p), "."))

hiddenimports = collect_submodules("PySide6")

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

a = Analysis(
    [str(ENTRY_SCRIPT)],
    pathex=[str(SPEC_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Choose appropriate icon for the platform if available
icon_map = {
    "win32": "VE_icon.ico",
    "darwin": "VE_icon.icns",
}
icon_file = None
mapped = icon_map.get(sys.platform)
if mapped and (SPEC_DIR / mapped).exists():
    icon_file = mapped
elif (SPEC_DIR / "VE_icon.png").exists():
    icon_file = "VE_icon.png"

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=icon_file,
)

# macOS: wrap executable into .app bundle
if sys.platform == "darwin":
    target = BUNDLE(
        exe,
        name=f"{APP_NAME}.app",
        icon=icon_file,
        bundle_identifier="com.example.virtualelevationrecipes",
    )
else:
    target = exe

coll = COLLECT(
    target,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)
