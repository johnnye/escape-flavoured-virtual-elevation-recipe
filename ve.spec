# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for the VirtualElevation desktop application
=================================================================
A *single* spec that works on Windows, macOS and Linux.
Usage (run on each OS):

    pyinstaller ve.spec --onefile --windowed --noconfirm

The GitHub Actions workflow (build.yaml) already runs this command.

Things to tweak if you add new resources:
  • DATA_FOLDERS – any folder that must ship with the binary
  • hiddenimports – add modules PyInstaller can’t detect automatically
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None
APP_NAME = "VirtualElevationRecipes"

# -----------------------------------------------------------------------------
# Data files – entire folders get bundled as‑is under the same relative path
# -----------------------------------------------------------------------------
DATA_FOLDERS = ["ui", "models", "utils", "config"]

datas = [(folder, folder) for folder in DATA_FOLDERS]

# Ship all three icon variants; the platform‑specific one is selected below
for icon in ("VE_icon.png", "VE_icon.ico", "VE_icon.icns"):
    datas.append((icon, "."))

# -----------------------------------------------------------------------------
# Hidden imports – Qt plugins & anything PyInstaller misses
# -----------------------------------------------------------------------------
hiddenimports = collect_submodules("PySide6")

# -----------------------------------------------------------------------------
# Analysis – entry point is main.py in the project root
# -----------------------------------------------------------------------------
a = Analysis(
    ["main.py"],
    pathex=[str(Path(__file__).parent)],
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

# Zip the pure‑Python bytecode
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Pick the right icon for the current build platform
icon_file = {
    "win32": "VE_icon.ico",
    "darwin": "VE_icon.icns",
}.get(sys.platform, "VE_icon.png")

# Build the executable (one‑file is toggled via the CLI flag)
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
    console=False,      # --windowed flag hides the console on Windows
    icon=icon_file,
)

# Collect everything into the final dist/VirtualElevation (or .exe, .app, etc.)
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
