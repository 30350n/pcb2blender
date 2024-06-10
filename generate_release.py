#!/usr/bin/env python3

import json, re, shutil, sys
from hashlib import sha256
from itertools import chain
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import requests
from autopep8 import main as autopep8
from git import Repo
from pytest import main as pytest

from _error_helper import *

RELEASE_DIRECTORY = Path("release")
ARCHIVE_DIRECTORY = RELEASE_DIRECTORY / "archive"

FILE_DIR = Path(__file__).parent
KICAD_ADDON_DIR = FILE_DIR / "pcb2blender_exporter"
BLENDER_ADDON_DIR = FILE_DIR / "pcb2blender_importer"

def generate_release(skip_tests=False, ignore_git_issues=False):
    info("running autopep8 ...")
    autopep8(["", "--recursive", "--in-place", "."])

    repo = Repo()

    if not ignore_git_issues:
        if repo.is_dirty():
            return error("repo is dirty (stash changes before generating a release)")

        if repo.head.is_detached:
            return error("repo is in detached head state")
        if "up to date" not in repo.git.status():
            return error("current commit is not pushed")

    tags = list(reversed(sorted(repo.tags, key=lambda tag: tag.commit.committed_datetime)))
    if tags[0].commit != repo.commit():
        return error("current commit is not tagged")

    if not skip_tests:
        with patch.object(sys, "argv", ["", "-q"]):
            info("running pytest ...")
            if (exit_code := pytest()) != 0:
                return error(f"tests failed with {exit_code}")

    info(f"generating release for {tags[0]} ... ")

    ARCHIVE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for path in RELEASE_DIRECTORY.glob("*.zip*"):
        if not path.is_file():
            continue
        shutil.move(path, ARCHIVE_DIRECTORY / path.name)

    generate_kicad_addon(
        KICAD_ADDON_DIR,
        METADATA,
        tags,
        "images/icon.png",
        ["images/blender_icon_32x32.png"],
    )

    generate_blender_addon(BLENDER_ADDON_DIR, tags)

    success("generated release!")

def generate_kicad_addon(path, metadata, tags, icon_path=None, extra_files=[]):
    version, kicad_version, _ = tags[0].name.split("-")

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
            "kicad_version": kicad_version[1:],
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
    download_url = f"{ORIGIN}/releases/download/{tags[0].name}/{zip_path.name}"
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

def generate_blender_addon(path, tags, extra_files=[]):
    version, _, blender_version = tags[0].name.split("-")

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

If you want to support this project, you can do so at https://github.com/sponsors/30350n
""").replace("\n ", "\n").strip()

METADATA_CONTACT = {
    "name": "Bobbe",
    "contact": {
        "web": "https://30350n.de/",
        "github": "https://github.com/30350n",
        "discord": "30350n"
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

if __name__ == "__main__":
    skip_tests = "--skip-tests" in sys.argv
    ignore_git = "--ignore-git" in sys.argv
    generate_release(skip_tests=skip_tests, ignore_git_issues=ignore_git)
