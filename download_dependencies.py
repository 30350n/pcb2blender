#!/usr/bin/env python3

import re
import shutil
from pathlib import Path
from subprocess import STDOUT, CalledProcessError, check_output
from typing import Any, cast

import tomlkit
from error_helper import warning

FILE_DIR = Path(__file__).parent
PYPROJECT_TOML = FILE_DIR / "pyproject.toml"
BLENDER_ADDON_DIR = FILE_DIR / "pcb2blender_importer"
BLENDER_ADDON_MANIFEST_TOML = BLENDER_ADDON_DIR / "blender_manifest.toml"
BLENDER_ADDON_WHEEL_DIR = BLENDER_ADDON_DIR / "wheels"

BLENDER_ADDON_SUPPORTED_PLATFORMS = {
    "windows-x64": ("win_amd64",),
    "windows-arm64": ("win_arm64",),
    "linux-x64": ("manylinux_2_28_x86_64",),
    "macos-x64": ("macosx_10_10_x86_64", "macosx_11_0_x86_64"),
    "macos-arm64": ("macosx_11_0_arm64",),
}
BLENDER_ADDON_SUPPORTED_PYTHON_VERSIONS = ("3.13", "3.14")

DOWNLOAD_BASE_COMMAND = (
    "pip",
    "download",
    *("--dest", BLENDER_ADDON_WHEEL_DIR),
    *("--only-binary", ":all:"),
)


def download_blender_extension_dependencies():
    if BLENDER_ADDON_WHEEL_DIR.is_dir():
        shutil.rmtree(BLENDER_ADDON_WHEEL_DIR)
    BLENDER_ADDON_WHEEL_DIR.mkdir()

    with PYPROJECT_TOML.open() as file:
        pyproject = cast(dict[str, Any], tomlkit.load(file))
        dependencies: list[str] = pyproject["project"]["dependencies"]

    for dependency in dependencies:
        for python_version in BLENDER_ADDON_SUPPORTED_PYTHON_VERSIONS:
            for blender_platform, wheel_platforms in BLENDER_ADDON_SUPPORTED_PLATFORMS.items():
                try:
                    check_output(
                        DOWNLOAD_BASE_COMMAND
                        + ("--python-version", python_version)
                        + sum(zip(("--platform",) * len(wheel_platforms), wheel_platforms), ())
                        + (dependency,),
                        stderr=STDOUT,
                    )
                except CalledProcessError as e:
                    warning(
                        f"failed to download '{dependency}' for Python {python_version} "
                        f"on '{blender_platform}' with:"
                    )
                    warning(e.output.decode().replace("\n", "\n  "), prefix="  ")
                    continue

                dependency_name = re.sub(
                    r"[-_.]+", "_", dependency.split("==", maxsplit=1)[0]
                ).lower()
                dependency_wheels = (
                    filepath
                    for filepath in BLENDER_ADDON_WHEEL_DIR.glob("*.whl")
                    if re.sub(r"[-_.]+", "_", filepath.name.split("-", maxsplit=1)[0]).lower()
                    == dependency_name
                )
                if any(filepath.name.endswith("-any.whl") for filepath in dependency_wheels):
                    break

    for numpy_wheel in BLENDER_ADDON_WHEEL_DIR.glob("numpy-*.whl"):
        numpy_wheel.unlink()

    blender_manifest = tomlkit.loads(BLENDER_ADDON_MANIFEST_TOML.read_text())
    blender_manifest["wheels"] = [
        f"./wheels/{filepath.name}" for filepath in BLENDER_ADDON_WHEEL_DIR.glob("*.whl")
    ]
    BLENDER_ADDON_MANIFEST_TOML.write_text(tomlkit.dumps(blender_manifest), newline="\n")


if __name__ == "__main__":
    download_blender_extension_dependencies()
