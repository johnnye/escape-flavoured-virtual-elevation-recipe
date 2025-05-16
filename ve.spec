# -*- mode: python ; coding: utf-8 -*-
"""
ve.spec – PyInstaller recipe for *VirtualElevationRecipes*

It tries these two entry-point candidates in order:
  1) <repo root>/main.py
  2) <repo root>/escape-flavoured-virtual-elevation-recipe/main.py

Adjust ENTRY_CANDIDATES if your structure changes.
Run with:  pyinstaller ve.spec --clean --noconfirm
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

APP_NAME = "VirtualElevationRecipes"
block_cipher = None

# ---------------------------------------------------------------------------
# Where are we?
# ---------------------------------------------------------------------------
SPEC_DIR = Path(globals().get("_specfile", Path.cwd())).parent

# Possible locations of the real start-up script
ENTRY_CANDIDATES = [
    SPEC_DIR / "main.py",
    SPEC_DIR / "escape-flavoured-virtual-elevation-recipe" / "main.py",
]

for candidate in ENTRY_CANDIDATES:
    if candidate.exists():
        ENTRY_SCRIPT = candidate
        SRC_ROOT = candidate.parent            # use this for data folders
        break
else:
    raise SystemExit(
        "❌  Could not find main.py in any of the expected locations:\n"
        + "\n".join([str(p) for p in ENTRY_CANDIDATES])
    )

# ---------------------------------------------------------------------------
# Resources – copy only what exists
# ---------------------------------------------------------------------------
DATA_FOLDERS = ["ui", "models", "utils", "config"]
datas = [
    (str(SRC_ROOT / f), f) for f in DATA_FOLDERS if (SRC_ROOT / f).exists()
]

for icon in ("VE_icon.png", "VE_icon.ico", "VE_icon.icns"):
    p = SPEC_DIR / icon
    if p.exists():
        datas.append((str(p), "."))

# ---------------------------------------------------------------------------
# Hidden imports – Qt plugins & dynamic modules
# ---------------------------------------------------------------------------
hiddenimports = collect_submodules("PySide6")

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
a = Analysis(
    [str(ENTRY_SCRIPT)],
    pathex=[str(SPEC_DIR), str(SRC_ROOT)],
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

icon_file = {
    "win32": "VE_icon.ico",
    "darwin": "VE_icon.icns",
}.get(sys.platform, "VE_icon.png") if (SPEC_DIR / "VE_icon.png").exists() else None

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