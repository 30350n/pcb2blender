import re
import shutil
import sys
import tarfile
from pathlib import Path

import requests


def download_blender(version: str):
    version = version.split("lts")[0]
    response: requests.Response = requests.get(f"{BASE_URL}{version}/")
    assert isinstance(response.content, bytes)

    versions = [match.group(1) for match in LINUX_VERSION_REGEX.finditer(response.content.decode())]
    latest = sorted(versions)[-1]

    download_path = Path(latest)
    download: requests.Response = requests.get(f"{BASE_URL}{version}/{latest}", stream=True)
    with download_path.open("wb") as file:
        shutil.copyfileobj(download.raw, file)  # pyright: ignore[reportArgumentType]
    with tarfile.open(Path(download_path), mode="r:xz") as file:
        file.extractall()
    download_path.unlink()

    blender_dir = next((path for path in Path().glob("blender-*")))
    blender_dir.rename("blender")


BASE_URL = "https://download.blender.org/release/Blender"
LINUX_VERSION_REGEX = re.compile('<a href="([^"]*.tar.xz)">')

if __name__ == "__main__":
    if len(sys.argv) == 2:
        download_blender(sys.argv[1].strip())
    else:
        print("./download_blender.py <version>")
