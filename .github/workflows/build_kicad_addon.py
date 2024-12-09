#!/usr/bin/env python3

import json
from argparse import ArgumentParser
from hashlib import sha256
from itertools import chain
from pathlib import Path
from subprocess import check_output
from zipfile import ZIP_DEFLATED, ZipFile

import requests

def build_kicad_addon(
    release_tag: str,
    path: Path = Path(),
    output_path: Path = Path(),
    icon: Path | None = None,
    extra_files: list[Path] = [],
):
    metadata: dict = json.loads((path / "metadata.json").read_text())

    version_str = metadata["versions"][0]["version"].replace(".", "-")
    kicad_version_str = metadata["versions"][0]["kicad_version"].replace(".", "-")
    zip_file_path = output_path / f"{path.name}_v{version_str}_k{kicad_version_str}.zip"
    with ZipFile(zip_file_path, mode="w", compression=ZIP_DEFLATED) as zip_file:
        plugin_dir = Path("plugins")

        extra_paths = (path / extra_file for extra_file in extra_files)
        for filepath in chain(path.glob("**/*.py"), extra_paths):
            zip_file.write(filepath, str(plugin_dir / filepath.relative_to(path)))

        if icon:
            zip_file.write(path / icon, "resources/icon.png")

        zip_file.writestr("metadata.json", json.dumps(metadata, indent=4))

    hash_file_path = Path(f"{zip_file_path.name}.sha256")
    with open(zip_file_path, "rb") as file:
        zip_file_hash = sha256(file.read()).hexdigest()
        hash_file_path.write_text(zip_file_hash)

    content_library_metadata = metadata.copy()

    version_metadata: dict = metadata["versions"][0].copy()
    version_metadata["download_sha256"] = zip_file_hash
    repo_url = get_repo_url(path)
    download_url = f"{repo_url}/releases/download/{release_tag}/{zip_file_path}"
    version_metadata["download_url"] = download_url
    with ZipFile(zip_file_path, mode="r") as zip_file:
        version_metadata["download_size"] = sum(
            (info.compress_size for info in zip_file.infolist())
        )
        version_metadata["install_size"] = sum(
            (info.file_size for info in zip_file.infolist())
        )

    content_library_metadata["versions"] = [version_metadata]

    other_tags = get_repo_tags(path)
    other_tags.remove(release_tag)

    for tag in other_tags:
        url = f"{repo_url}/releases/download/{tag}/version.json"
        result = requests.get(url)
        if result.ok:
            content_library_metadata["versions"].append(result.json())
        else:
            print(f"hint: skipping {tag}, missing version.json")

    version_metadata_json = json.dumps(version_metadata, indent=4)
    (output_path / "version.json").write_text(version_metadata_json)

    content_library_metadata_json = json.dumps(content_library_metadata, indent=4)
    (output_path / "metadata.json").write_text(content_library_metadata_json)

def get_repo_url(path: Path):
    command = ("gh", "repo", "view", "--json", "url", "--template", "{{.url}}")
    return check_output(command, cwd=path).decode()

def get_repo_tags(path: Path):
    command = (
        "gh", "release", "list", "--json", "tagName", "--template",
        "{{range .}}{{.tagName}}{{\" \"}}{{end}}"
    )
    return check_output(command, cwd=path).decode().split()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("release_tag")
    parser.add_argument("--source", default="", help="addon source directory")
    parser.add_argument("--out", default="", help="output directory")
    parser.add_argument("--icon", default="", help="path to addon icon (relative to SOURCE)")
    parser.add_argument(
        "--extra-files", nargs="*", default=[],
        help="path to extra addon files (relative to SOURCE)"
    )
    args = parser.parse_args()

    build_kicad_addon(
        args.release_tag,
        Path(args.source),
        Path(args.out),
        Path(args.icon),
        [Path(extra_file) for extra_file in args.extra_files]
    )
