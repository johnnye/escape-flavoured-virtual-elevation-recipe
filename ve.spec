# -*- mode: python ; coding: utf-8 -*-
"""ve.spec – PyInstaller build recipe for **VirtualElevationRecipes**

Invoked by the GitHub Actions workflow with:

    pyinstaller ve.spec --clean --noconfirm

Produces an **onedir** bundle named `VirtualElevationRecipes` on each OS.
The workflow post‑processes it into a dmg (macOS) or AppImage (Linux) and
zips it for all platforms.  Adjust DATA_FOLDERS if you add new resource
sub‑directories.
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None
APP_NAME = "VirtualElevationRecipes"

# ---------------------------------------------------------------------------
# Resolve spec location – PyInstaller 6 sets _specfile; fall back to CWD.
# ---------------------------------------------------------------------------
SPEC_DIR = Path(globals().get("_specfile", Path.cwd())).parent

# ---------------------------------------------------------------------------
# Data files – bundle these folders verbatim alongside the executable.
# ---------------------------------------------------------------------------
DATA_FOLDERS = ["ui", "models", "utils", "config"]

datas = [(str(SPEC_DIR / folder), folder) for folder in DATA_FOLDERS]

# Icons: supply all three formats so each OS picks its native one.
for icon in ("VE_icon.png", "VE_icon.ico", "VE_icon.icns"):
    datas.append((str(SPEC_DIR / icon), "."))

# ---------------------------------------------------------------------------
# Hidden imports – Qt plugins and anything PyInstaller misses.
# ---------------------------------------------------------------------------
hiddenimports = collect_submodules("PySide6")

# ---------------------------------------------------------------------------
# Analysis – specify entry‑point and resources.
# ---------------------------------------------------------------------------
a = Analysis(
    [str(SPEC_DIR / "main.py")],
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

# Choose icon for current platform
icon_file = {
    "win32": "VE_icon.ico",
    "darwin": "VE_icon.icns",
}.get(sys.platform, "VE_icon.png")

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
    console=False,  # GUI app – hide console window
    icon=icon_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)
