# -*- mode: python ; coding: utf-8 -*-
"""ve.spec – PyInstaller recipe for **VirtualElevationRecipes**

* Builds from the real entry‑point at
    escape-flavoured-virtual-elevation-recipe/main.py
* Produces an **onedir** bundle named `VirtualElevationRecipes` (workflow
  later wraps it into a dmg/AppImage and zips it for distribution).

Invoke via the CI pipeline or locally:

    pyinstaller ve.spec --clean --noconfirm
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None
APP_NAME = "VirtualElevationRecipes"

# ---------------------------------------------------------------------------
# Locate source tree & entry script
# ---------------------------------------------------------------------------
SPEC_DIR = Path(globals().get("_specfile", Path.cwd())).parent
SRC_DIR = SPEC_DIR / "escape-flavoured-virtual-elevation-recipe"
ENTRY_SCRIPT = SRC_DIR / "main.py"

# ---------------------------------------------------------------------------
# Bundle resource folders located *inside* the source tree
# ---------------------------------------------------------------------------
DATA_FOLDERS = ["ui", "models", "utils", "config"]

datas = [(str(SRC_DIR / folder), folder) for folder in DATA_FOLDERS]

# Icons reside next to the spec
for icon in ("VE_icon.png", "VE_icon.ico", "VE_icon.icns"):
    datas.append((str(SPEC_DIR / icon), "."))

# ---------------------------------------------------------------------------
# Hidden imports – ensure Qt plugins & dynamic modules are included
# ---------------------------------------------------------------------------
hiddenimports = collect_submodules("PySide6")

# ---------------------------------------------------------------------------
# Analysis phase – feed PyInstaller the entry script & resources
# ---------------------------------------------------------------------------
a = Analysis(
    [str(ENTRY_SCRIPT)],
    pathex=[str(SPEC_DIR), str(SRC_DIR)],
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
