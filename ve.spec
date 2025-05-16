# -*- mode: python ; coding: utf-8 -*-
"""ve.spec – PyInstaller recipe for **VirtualElevationRecipes**

Assumptions about repo layout (root = directory containing this spec):

    ├── ve.spec
    ├── main.py               # application entry point
    ├── VE_icon.png|ico|icns   # icons (at least one)
    ├── ui/
    ├── models/
    ├── utils/
    └── config/

If any of those folders are missing, just remove them from DATA_FOLDERS.
Invocation (CI or local):

    pyinstaller ve.spec --clean --noconfirm

This builds an **onedir** bundle named `VirtualElevationRecipes` in ./dist.
The CI workflow then post‑processes it (zip, dmg, AppImage).
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None
APP_NAME = "VirtualElevationRecipes"

# Root of the repository – directory where ve.spec resides
SPEC_DIR = Path(globals().get("_specfile", Path.cwd())).parent

# Entry script
ENTRY_SCRIPT = SPEC_DIR / "main.py"
if not ENTRY_SCRIPT.exists():
    raise SystemExit(f"Entry script not found: {ENTRY_SCRIPT}")

# Resource folders to copy verbatim (skip if they don’t exist)
DATA_FOLDERS = ["ui", "models", "utils", "config"]

datas = []
for folder in DATA_FOLDERS:
    src = SPEC_DIR / folder
    if src.exists():
        datas.append((str(src), folder))

# Icons – add whichever variants actually exist
for icon in ("VE_icon.png", "VE_icon.ico", "VE_icon.icns"):
    p = SPEC_DIR / icon
    if p.exists():
        datas.append((str(p), "."))

# Hidden imports – Qt plugins & modules PyInstaller may miss
hiddenimports = collect_submodules("PySide6")

# Analysis: instruct PyInstaller where to start
pathex = [str(SPEC_DIR)]

a = Analysis(
    [str(ENTRY_SCRIPT)],
    pathex=pathex,
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

# Choose platform‑native icon if we bundled one
icon_map = {
    "win32": "VE_icon.ico",
    "darwin": "VE_icon.icns",
}
icon_file = next((i for k, i in icon_map.items() if sys.platform == k and (SPEC_DIR / i).exists()), None)
if icon_file is None and (SPEC_DIR / "VE_icon.png").exists():
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
