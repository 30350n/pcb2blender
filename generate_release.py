#!/usr/bin/env python3

import ast, json, re, shutil, sys
from hashlib import sha256
from itertools import chain
from pathlib import Path
from subprocess import DEVNULL, CalledProcessError, check_call
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import requests, toml
from autopep8 import main as autopep8
from error_helper import *
from git import Repo
from pytest import main as pytest

RELEASE_DIRECTORY = Path("release")
ARCHIVE_DIRECTORY = RELEASE_DIRECTORY / "archive"
TEMP_WHEEL_DIRECTORY = RELEASE_DIRECTORY / "wheels_tmp"

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
    generate_blender_extension(BLENDER_ADDON_DIR, tags)

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

def generate_blender_addon(path, tags):
    version, _, blender_version = tags[0].name.split("-")

    if version[1:] != (package_version := get_package_version(path)):
        warning(f"tag addon version '{version[1:]}' doesn't match package version "
                f"'{package_version}'")

    if version[1:] != (bl_info_version := ".".join(get_bl_info_version(path)[:2])):
        warning(f"tag addon version '{version[1:]}' doesn't match addon version "
                f"in bl_info '{bl_info_version}'")

    if blender_version[1:] != (bl_info_bversion := ".".join(get_bl_info_bversion(path)[:2])):
        warning(f"tag blender version '{version[1:]}' doesn't match blender version "
                f"in bl_info '{bl_info_bversion}'")

    RELEASE_DIRECTORY.mkdir(exist_ok=True)
    zip_path = RELEASE_DIRECTORY / f"{path.name}_{version[1:].replace('.', '-')}.zip"
    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as zip_file:
        for filepath in path.glob("**/*.py"):
            if "site-packages" in str(filepath) or "docs" in str(filepath):
                continue
            zip_file.write(filepath, filepath.relative_to(path.parent))

    zip_hash_path = Path(f"{str(zip_path)}.sha256")
    with open(zip_path, "rb") as file:
        zip_hash = sha256(file.read()).hexdigest()
        zip_hash_path.write_text(zip_hash)

    metadata_dir = RELEASE_DIRECTORY / "metadata"
    metadata_dir.mkdir(exist_ok=True)

def generate_blender_extension(path, tags):
    version, _, blender_version = tags[0].name.split("-")

    if version[1:] != (package_version := get_package_version(path)):
        warning(f"tag addon version '{version[1:]}' doesn't match package version "
                f"'{package_version}'")

    if version[1:] != (bl_info_version := ".".join(get_bl_info_version(path)[:2])):
        warning(f"tag addon version '{version[1:]}' doesn't match addon version "
                f"in bl_info '{bl_info_version}'")

    if blender_version[1:] != (bl_info_bversion := ".".join(get_bl_info_bversion(path)[:2])):
        warning(f"tag blender version '{version[1:]}' doesn't match blender version "
                f"in bl_info '{bl_info_bversion}'")

    RELEASE_DIRECTORY.mkdir(exist_ok=True)

    init_content = (path / "__init__.py").read_text()
    dependencies = ast.literal_eval(extension_dependencies_regex.search(init_content).group(1))

    if TEMP_WHEEL_DIRECTORY.is_dir():
        shutil.rmtree(TEMP_WHEEL_DIRECTORY)
    TEMP_WHEEL_DIRECTORY.mkdir()

    for dependency in dependencies:
        for blender_platform, wheel_platforms in SUPPORTED_PLATFORMS.items():
            for wheel_platform in wheel_platforms:
                try:
                    check_call((
                        "pip", "download", dependency, f"--dest={TEMP_WHEEL_DIRECTORY}",
                        "--only-binary=:all:", f"--python-version={BLENDER_PYTHON}",
                        f"--platform={wheel_platform}", "--quiet"
                    ), stdout=DEVNULL, stderr=DEVNULL)
                    break
                except CalledProcessError:
                    continue
            else:
                warning(f"failed to download '{dependency}' for '{blender_platform}'")

    zip_path = RELEASE_DIRECTORY / f"{path.name}_extension_{version[1:].replace('.', '-')}.zip"
    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as zip_file:
        for filepath in path.glob("**/*.py"):
            if "site-packages" in str(filepath) or "docs" in str(filepath):
                continue
            if str(relative_path := filepath.relative_to(path)) == "__init__.py":
                continue
            zip_file.write(filepath, relative_path)
        zip_file.writestr("__init__.py", extension_init_patch.sub("", init_content))

        wheels = []
        for filepath in TEMP_WHEEL_DIRECTORY.glob("*.whl"):
            zip_file.write(filepath, f"wheels/{filepath.name}")
            wheels.append(f"./wheels/{filepath.name}")

        manifest = MANIFEST | {
            "version": f"{version[1:]}.0",
            "blender_version_min": ".".join(get_bl_info_bversion(path)),
            "wheels": wheels,
        }
        zip_file.writestr("blender_manifest.toml", toml.dumps(manifest))

    shutil.rmtree(TEMP_WHEEL_DIRECTORY)

    zip_hash_path = Path(f"{str(zip_path)}.sha256")
    with open(zip_path, "rb") as file:
        zip_hash = sha256(file.read()).hexdigest()
        zip_hash_path.write_text(zip_hash)

    metadata_dir = RELEASE_DIRECTORY / "metadata"
    metadata_dir.mkdir(exist_ok=True)

def get_bl_info_version(path):
    group = version_regex.search((path / "__init__.py").read_text()).groups()[0]
    return tuple(s.strip() for s in group.split(","))

def get_bl_info_bversion(path):
    group = bversion_regex.search((path / "__init__.py").read_text()).groups()[0]
    return tuple(s.strip() for s in group.split(","))

def get_package_version(path):
    return package_version_regex.search((path / "__init__.py").read_text()).groups()[0]

version_regex  = re.compile(r"bl_info\s*=\s*{[^}]*\"version\"\s*:\s*\(([^^\)]*)\)\s*,[^}]*}")
bversion_regex = re.compile(r"bl_info\s*=\s*{[^}]*\"blender\"\s*:\s*\(([^^\)]*)\)\s*,[^}]*}")
package_version_regex  = re.compile(r"__version__\s*=\s*\"([^\"]*)\"")

DESCRIPTION_SHORT = "Export PCB 3D Models from KiCad to Blender"
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
    "description": DESCRIPTION_SHORT,
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

BLENDER_PYTHON = "3.11"

SUPPORTED_PLATFORMS = {
    "windows-amd64": ("win_amd64",),
    "linux-x86_64": ("manylinux_2_28_x86_64", "manylinux_2_17_x86_64"),
    "macos-x86_64": ("macosx_10_10_x86_64", "macosx_10_9_x86_64"),
    "macos-arm64": ("macosx_11_0_arm64",),
}

MANIFEST = {
    "schema_version": "1.0.0",
    "id": "pcb2blender",
    "version": None,
    "name": "pcb2blender",
    "tagline": DESCRIPTION_SHORT,
    "maintainer": "30350n (Max Schlecht)",
    "type": "add-on",
    "permissions": ["files"],
    "website": ORIGIN,
    "tags": ["Import-Export"],
    "blender_version_min": None,
    "license": ["SPDX:GPL-3.0-or-later"],
    "platforms": list(SUPPORTED_PLATFORMS.keys()),
    "wheels": None,
}

extension_dependencies_regex = re.compile(r"deps\s*=\s*({[^}]*})")
extension_init_patch = re.compile(
    r"(add_dependencies(?:,\s+))|((?:,\s+)add_dependencies)|(add_dependencies(.*)\s)"
    r"|(\sdeps\s*=\s*{[^}]*}\s)|(bl_info\s*=\s*{[^}]*}\s*)"
)

if __name__ == "__main__":
    skip_tests = "--skip-tests" in sys.argv
    ignore_git = "--ignore-git" in sys.argv
    generate_release(skip_tests=skip_tests, ignore_git_issues=ignore_git)
