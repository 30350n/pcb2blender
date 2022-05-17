#!/usr/bin/env python3

from pathlib import Path
from git import Repo
from zipfile import ZipFile, ZIP_DEFLATED
from hashlib import sha256
from itertools import chain
import json, shutil, requests

TARGET_DIRECTORY = Path("release")

DESCRIPTION = """
The pcb2blender workflow lets you create professionally looking product renders of all your
KiCad projects in minutes! Simply export your board as a .pcb3d file in KiCad, import it into
Blender and start creating!
""".replace("\n", " ")

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

def generate_kicad_addon(path, metadata, icon_path=None, extra_files=[]):
    repo = Repo()
    tags = list(reversed(sorted(repo.tags, key=lambda tag: tag.commit.committed_datetime)))

    latest_tag = tags[0]
    latest_version, latest_kicad_version, _ = latest_tag.name.split("-")

    TARGET_DIRECTORY.mkdir(exist_ok=True)
    zip_path = TARGET_DIRECTORY / f"{path.name}_{latest_version[1:].replace('.', '-')}.zip"
    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as zip_file:
        plugin_dir = Path("plugins")
        for filepath in path.glob("**/*.py"):
            zip_file.write(filepath, str(plugin_dir / filepath.relative_to(path)))

        for filepath in extra_files:
            zip_file.write(path / filepath, plugin_dir / filepath)

        if icon_path:
            zip_file.write(path / icon_path, "resources/icon.png")

        metadata_latest_version = {
            "version": latest_version[1:],
            "status": "stable",
            "kicad_version": latest_kicad_version[1:]
        }

        metadata["versions"] = [metadata_latest_version]
        metadata_json = json.dumps(metadata, indent=4)
        zip_file.writestr("metadata.json", metadata_json)

    zip_hash_path = Path(f"{str(zip_path)}.sha256")
    with open(zip_path, "rb") as file:
        zip_hash = sha256(file.read()).hexdigest()
        zip_hash_path.write_text(zip_hash)

    metadata_dir = TARGET_DIRECTORY / "metadata"
    metadata_dir.mkdir(exist_ok=True)

    if icon_path:
        shutil.copy((path / icon_path), metadata_dir / "icon.png")

    metadata_latest_version["download_sha256"] = zip_hash
    download_url = f"{ORIGIN}/releases/download/{latest_tag.name}/{zip_path.name}"
    metadata_latest_version["download_url"] = download_url
    with ZipFile(zip_path, mode="r") as zip_file:
        metadata_latest_version["download_size"] = sum(
            (info.compress_size for info in zip_file.infolist()))
        metadata_latest_version["install_size"] = sum(
            (info.file_size for info in zip_file.infolist()))

    version_json = json.dumps(metadata_latest_version, indent=4)
    (TARGET_DIRECTORY / "version.json").write_text(version_json)

    for tag in tags[1:]:
        url = f"{ORIGIN}/releases/download/{tag.name}/version.json"
        metadata["versions"].append(requests.get(url).json())

    metadata_json = json.dumps(metadata, indent=4)
    (metadata_dir / "metadata.json").write_text(metadata_json)

def generate_blender_addon(path, extra_files=[]):
    repo = Repo()
    tags = list(reversed(sorted(repo.tags, key=lambda tag: tag.commit.committed_datetime)))

    latest_tag = tags[0]
    version, _, blender_version = latest_tag.name.split("-")

    TARGET_DIRECTORY.mkdir(exist_ok=True)
    zip_path = TARGET_DIRECTORY / f"{path.name}_{version[1:].replace('.', '-')}.zip"
    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as zip_file:
        extra_paths = (path / extra for extra in extra_files)
        for filepath in chain(path.glob("**/*.py"), extra_paths):
            if "site-packages" in str(filepath):
                continue
            zip_file.write(filepath, filepath.relative_to(path.parent))

    zip_hash_path = Path(f"{str(zip_path)}.sha256")
    with open(zip_path, "rb") as file:
        zip_hash = sha256(file.read()).hexdigest()
        zip_hash_path.write_text(zip_hash)

    metadata_dir = TARGET_DIRECTORY / "metadata"
    metadata_dir.mkdir(exist_ok=True)

if __name__ == "__main__":
    generate_kicad_addon(
        Path(__file__).parent / "pcb2blender_exporter",
        METADATA,
        "images/icon.png",
        ["images/blender_icon_32x32.png"],
    )

    generate_blender_addon(
        Path(__file__).parent / "pcb2blender_importer",
    )
