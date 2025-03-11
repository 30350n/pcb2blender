#!/usr/bin/env python3

import shutil
from pathlib import Path
from subprocess import DEVNULL, CalledProcessError, check_call

import tomlkit
from error_helper import *

FILE_DIR = Path(__file__).parent
BLENDER_ADDON_DIR = FILE_DIR / "pcb2blender_importer"
BLENDER_ADDON_MANIFEST_TOML = BLENDER_ADDON_DIR / "blender_manifest.toml"
BLENDER_ADDON_WHEEL_DIR = BLENDER_ADDON_DIR / "wheels"

BLENDER_PYTHON_VERSION = "3.11"

BLENDER_ADDON_PYTHON_DEPENDENCIES = [
    "error-helper",
    "skia-python",
    "pillow",
]

BLENDER_ADDON_SUPPORTED_PLATFORMS = {
    "windows-x64": ("win_amd64",),
    "linux-x64": ("manylinux_2_28_x86_64", "manylinux_2_17_x86_64"),
    "macos-x64": ("macosx_10_10_x86_64", "macosx_10_9_x86_64"),
    "macos-arm64": ("macosx_11_0_arm64",),
}

DOWNLOAD_BASE_COMMAND = (
    "pip",
    "download",
    "--quiet",
    *("--dest", BLENDER_ADDON_WHEEL_DIR),
    *("--only-binary", ":all:"),
    *("--python-version", BLENDER_PYTHON_VERSION),
)


def download_blender_extension_dependencies():
    if BLENDER_ADDON_WHEEL_DIR.is_dir():
        shutil.rmtree(BLENDER_ADDON_WHEEL_DIR)
    BLENDER_ADDON_WHEEL_DIR.mkdir()

    for dependency in BLENDER_ADDON_PYTHON_DEPENDENCIES:
        for blender_platform, wheel_platforms in BLENDER_ADDON_SUPPORTED_PLATFORMS.items():
            for wheel_platform in wheel_platforms:
                try:
                    command = DOWNLOAD_BASE_COMMAND + ("--platform", wheel_platform, dependency)
                    check_call(command, stdout=DEVNULL, stderr=DEVNULL)
                    break
                except CalledProcessError:
                    continue
            else:
                warning(f"failed to download '{dependency}' for '{blender_platform}'")

    for numpy_wheel in BLENDER_ADDON_WHEEL_DIR.glob("numpy-*.whl"):
        numpy_wheel.unlink()

    blender_manifest = tomlkit.parse(BLENDER_ADDON_MANIFEST_TOML.read_text())
    blender_manifest["wheels"] = [
        f"./wheels/{filepath.name}" for filepath in BLENDER_ADDON_WHEEL_DIR.glob("*.whl")
    ]
    BLENDER_ADDON_MANIFEST_TOML.write_text(tomlkit.dumps(blender_manifest), newline="\n")


if __name__ == "__main__":
    download_blender_extension_dependencies()
