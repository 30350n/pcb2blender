#!/usr/bin/env python3

from pathlib import Path
from git import Repo
from zipfile import ZipFile, ZIP_DEFLATED
from hashlib import sha256
from itertools import chain
from autopep8 import main as autopep8
from pytest import main as pytest
from unittest.mock import patch
import json, shutil, requests, re, sys

RELEASE_DIRECTORY = Path("release")
ARCHIVE_DIRECTORY = RELEASE_DIRECTORY / "archive"

def generate_release():
    info("running autopep8 ... ", end="")
    autopep8(["", "--recursive", "--in-place", "."])
    if Repo().is_dirty():
        error("repo is dirty (stash changes before generating a release)")

    with patch.object(sys, "argv", ["", "-q"]):
        info("running pytest ...")
        if (exit_code := pytest()) != 0:
            error(f"tests failed with {exit_code}")

    if Repo().head.is_detached:
        error("repo is in detached head state")
    if "up to date" not in Repo().git.status():
        error("current commit is not pushed")
    if Repo().tags[-1].commit != Repo().commit():
        error("current commit is not tagged")

    info(f"generating release for {Repo().tags[-1]} ... ")

    ARCHIVE_DIRECTORY.mkdir(exist_ok=True)
    for path in RELEASE_DIRECTORY.glob("*.zip*"):
        if not path.is_file():
            continue
        shutil.move(path, ARCHIVE_DIRECTORY / path.name)

    generate_kicad_addon(
        Path(__file__).parent / "pcb2blender_exporter",
        METADATA,
        "images/icon.png",
        ["images/blender_icon_32x32.png"],
    )

    generate_blender_addon(
        Path(__file__).parent / "pcb2blender_importer",
    )

    success("done.")

def generate_kicad_addon(path, metadata, icon_path=None, extra_files=[]):
    repo = Repo()
    tags = list(reversed(sorted(repo.tags, key=lambda tag: tag.commit.committed_datetime)))

    latest_tag = tags[0]
    version, kicad_version, _ = latest_tag.name.split("-")

    if version[1:] != (package_version := get_package_version(path)):
        warning(f"tag addon version '{version[1:]}' doesn't match package version "
                f"'{package_version}'")

    RELEASE_DIRECTORY.mkdir(exist_ok=True)
    zip_path = RELEASE_DIRECTORY / f"{path.name}_{version[1:].replace('.', '-')}.zip"
    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as zip_file:
        plugin_dir = Path("plugins")
        extra_paths = (path / extra for extra in extra_files)
        for filepath in chain(path.glob("**/*.py"), extra_paths):
            zip_file.write(filepath, str(plugin_dir / filepath.relative_to(path)))

        if icon_path:
            zip_file.write(path / icon_path, "resources/icon.png")

        metadata_version = {
            "version": version[1:],
            "status": "stable",
            "kicad_version": kicad_version[1:]
        }

        metadata["versions"] = [metadata_version]
        metadata_json = json.dumps(metadata, indent=4)
        zip_file.writestr("metadata.json", metadata_json)

    zip_hash_path = Path(f"{str(zip_path)}.sha256")
    with open(zip_path, "rb") as file:
        zip_hash = sha256(file.read()).hexdigest()
        zip_hash_path.write_text(zip_hash)

    metadata_dir = RELEASE_DIRECTORY / "metadata"
    metadata_dir.mkdir(exist_ok=True)

    if icon_path:
        shutil.copy((path / icon_path), metadata_dir / "icon.png")

    metadata_version["download_sha256"] = zip_hash
    download_url = f"{ORIGIN}/releases/download/{latest_tag.name}/{zip_path.name}"
    metadata_version["download_url"] = download_url
    with ZipFile(zip_path, mode="r") as zip_file:
        metadata_version["download_size"] = sum(
            (info.compress_size for info in zip_file.infolist()))
        metadata_version["install_size"] = sum(
            (info.file_size for info in zip_file.infolist()))

    version_json = json.dumps(metadata_version, indent=4)
    (RELEASE_DIRECTORY / "version.json").write_text(version_json)

    for tag in tags[1:]:
        url = f"{ORIGIN}/releases/download/{tag.name}/version.json"
        result = requests.get(url)
        if result.ok:
            metadata["versions"].append(result.json())
        else:
            hint(f"skipping {tag.name}, missing version.json")

    metadata_json = json.dumps(metadata, indent=4)
    (metadata_dir / "metadata.json").write_text(metadata_json)

def generate_blender_addon(path, extra_files=[]):
    repo = Repo()
    tags = list(reversed(sorted(repo.tags, key=lambda tag: tag.commit.committed_datetime)))

    latest_tag = tags[0]
    version, _, blender_version = latest_tag.name.split("-")

    if version[1:] != (package_version := get_package_version(path)):
        warning(f"tag addon version '{version[1:]}' doesn't match package version "
                f"'{package_version}'")

    if version[1:] != (bl_info_version := get_bl_info_version(path)):
        warning(f"tag addon version '{version[1:]}' doesn't match addon version "
                f"in bl_info '{bl_info_version}'")

    if blender_version[1:] != (bl_info_bversion := get_bl_info_bversion(path)):
        warning(f"tag blender version '{version[1:]}' doesn't match blender version "
                f"in bl_info '{bl_info_bversion}'")

    RELEASE_DIRECTORY.mkdir(exist_ok=True)
    zip_path = RELEASE_DIRECTORY / f"{path.name}_{version[1:].replace('.', '-')}.zip"
    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as zip_file:
        extra_paths = (path / extra for extra in extra_files)
        for filepath in chain(path.glob("**/*.py"), extra_paths):
            if "site-packages" in str(filepath) or "docs" in str(filepath):
                continue
            zip_file.write(filepath, filepath.relative_to(path.parent))

    zip_hash_path = Path(f"{str(zip_path)}.sha256")
    with open(zip_path, "rb") as file:
        zip_hash = sha256(file.read()).hexdigest()
        zip_hash_path.write_text(zip_hash)

    metadata_dir = RELEASE_DIRECTORY / "metadata"
    metadata_dir.mkdir(exist_ok=True)

def get_bl_info_version(path):
    group = version_regex.search((path / "__init__.py").read_text()).groups()[0]
    return ".".join(s.strip() for s in group.split(",")[:2])

def get_bl_info_bversion(path):
    group = bversion_regex.search((path / "__init__.py").read_text()).groups()[0]
    return ".".join(s.strip() for s in group.split(",")[:2])

def get_package_version(path):
    return package_version_regex.search((path / "__init__.py").read_text()).groups()[0]

version_regex  = re.compile(r"bl_info\s*=\s*{[^}]*\"version\"\s*:\s*\(([^^\)]*)\)\s*,[^}]*}")
bversion_regex = re.compile(r"bl_info\s*=\s*{[^}]*\"blender\"\s*:\s*\(([^^\)]*)\)\s*,[^}]*}")
package_version_regex  = re.compile(r"__version__\s*=\s*\"([^\"]*)\"")

DESCRIPTION = re.sub(r"\n([^\n])", " \\g<1>", """
The pcb2blender workflow lets you create professionally looking product renders of all your
KiCad projects in minutes! Simply export your board as a .pcb3d file in KiCad, import it into
Blender and start creating!

(Note for Linux/macOS users: If you run into any issues, make sure you're running python 3.10).
""").strip()

METADATA_CONTACT = {
    "name": "Bobbe",
    "contact": {
        "web": "https://30350n.de/",
        "github": "https://github.com/30350n",
        "discord": "Bobbe#8552"
    },
}

ORIGIN = "https://github.com/30350n/pcb2blender"

METADATA = {
    "$schema": "https://go.kicad.org/pcm/schemas/v1",
    "name": "pcb2blender",
    "description": "Export PCB 3D Models from Pcbnew to Blender",
    "description_full": DESCRIPTION,
    "identifier": "com.github.30350n.pcb2blender",
    "type": "plugin",
    "author": METADATA_CONTACT,
    "maintainer": METADATA_CONTACT,
    "license": "GPL-3.0",
    "resources": {
        "homepage": ORIGIN,
    },
}

COLOR_INFO = "\033[94m"
COLOR_HINT = "\033[2;3m"
COLOR_SUCCESS = "\033[92m"
COLOR_WARNING = "\033[93m"
COLOR_ERROR = "\033[91m"
COLOR_END = "\033[0m"

def info(msg, end="\n"):
    print(f"{COLOR_INFO}{msg}{COLOR_END}", end=end, flush=True)

def hint(msg):
    print(f"{COLOR_HINT}({msg}){COLOR_END}")

def success(msg):
    print(f"{COLOR_SUCCESS}{msg}{COLOR_END}")

def warning(msg):
    print(f"{COLOR_WARNING}warning: {msg}{COLOR_END}")

def error(msg):
    print(f"\n{COLOR_ERROR}error: {msg}{COLOR_END}", file=sys.stderr)
    exit()

if __name__ == "__main__":
    generate_release()
