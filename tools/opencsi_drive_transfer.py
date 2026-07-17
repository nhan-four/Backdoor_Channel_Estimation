from google.colab import auth
auth.authenticate_user()

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

import hashlib
import json
import pathlib
import time
import requests
from urllib.parse import quote

drive = build("drive", "v3")

ROOT_FOLDER_ID = "11TXh4knlWAZE1L7NHwemX421fp5dRZPK"
RAW_FOLDER_ID = "1VQQtjsBuj_kjqbYa1Slgc7OV75Mlw7bg"
CHECKSUMS_FOLDER_ID = "1FxxPxaUlq8ThJKO3dPsAMj040Mb_HRbn"
PAPER_FOLDER_ID = "1DxUCcFf9r7zUx0vEcuCwpaTtJbenZ1nG"
SOURCE_RESULTS_FOLDER_ID = "18HZV77QpQUhX9gxHlt7Ohps0Gyu5CLfp"

ARTICLE_ID = 19596379
EXPECTED_RAW_SIZE = 2_025_122_091
EXPECTED_RAW_MD5 = "7d83f1682d05fa230bba3e90f755c580"
EXPECTED_RAW_SHA256 = "3362474e5c00a0ab9a1993d2ed642de9c8503ec5048f58f8cbd96a639e4e92d0"

RELEASE_TAG = "opencsi-full-drive-transfer-v1"
RELEASE_BASE = (
    "https://github.com/nhan-four/Backdoor_Channel_Estimation/"
    f"releases/download/{RELEASE_TAG}"
)

WORK = pathlib.Path("/content/opencsi_drive_transfer")
WORK.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "OpenCSI-Repro-Transfer/1.0"})


def hash_file(path, algorithm="sha256", chunk_size=8 * 1024 * 1024):
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, target, expected_size=None, expected_sha256=None, retries=6):
    target = pathlib.Path(target)
    partial = pathlib.Path(str(target) + ".part")

    for attempt in range(1, retries + 1):
        downloaded = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
        mode = "ab" if downloaded else "wb"

        try:
            with SESSION.get(
                url,
                stream=True,
                timeout=(30, 300),
                headers=headers,
                allow_redirects=True,
            ) as response:
                if downloaded and response.status_code == 200:
                    downloaded = 0
                    mode = "wb"
                response.raise_for_status()
                total = expected_size or (
                    downloaded + int(response.headers.get("content-length", 0))
                )
                last_report = time.time()

                with open(partial, mode) as output:
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        if not chunk:
                            continue
                        output.write(chunk)
                        downloaded += len(chunk)
                        if time.time() - last_report >= 5:
                            percent = 100.0 * downloaded / total if total else 0.0
                            print(
                                f"  {target.name}: "
                                f"{downloaded / 1e6:.1f}/{total / 1e6:.1f} MB "
                                f"({percent:.1f}%)"
                            )
                            last_report = time.time()
            break
        except Exception as error:
            if attempt == retries:
                raise
            print(f"Retry {attempt}/{retries} for {target.name}: {error}")
            time.sleep(min(30, 2**attempt))

    if expected_size is not None and partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {target.name}: "
            f"{partial.stat().st_size} != {expected_size}"
        )

    if expected_sha256 is not None:
        actual_sha256 = hash_file(partial, "sha256")
        if actual_sha256.lower() != expected_sha256.lower():
            raise RuntimeError(
                f"SHA-256 mismatch for {target.name}: "
                f"{actual_sha256} != {expected_sha256}"
            )

    partial.replace(target)
    return target


def drive_find(parent_id, name):
    # All transfer filenames are fixed and contain no apostrophe.
    query = (
        f"'{parent_id}' in parents and "
        f"name = '{name}' and trashed = false"
    )
    response = drive.files().list(
        q=query,
        fields="files(id,name,size,md5Checksum,webViewLink)",
        pageSize=10,
    ).execute()
    return response.get("files", [])


def drive_upload(path, parent_id, mime_type="application/octet-stream"):
    path = pathlib.Path(path)

    for existing in drive_find(parent_id, path.name):
        if int(existing.get("size", -1)) == path.stat().st_size:
            print("SKIP existing same-size:", path.name)
            return existing

    media = MediaFileUpload(
        str(path),
        mimetype=mime_type,
        resumable=True,
        chunksize=100 * 1024 * 1024,
    )
    request = drive.files().create(
        body={"name": path.name, "parents": [parent_id]},
        media_body=media,
        fields="id,name,size,md5Checksum,webViewLink",
    )

    result = None
    while result is None:
        status, result = request.next_chunk(num_retries=10)
        if status:
            print(f"  Drive {path.name}: {status.progress() * 100:.1f}%")

    print("UPLOADED:", result["name"], result.get("webViewLink", result["id"]))
    return result


def get_figshare_raw():
    metadata_url = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
    response = SESSION.get(metadata_url, timeout=60)
    response.raise_for_status()
    metadata = response.json()

    candidates = [
        item
        for item in metadata.get("files", [])
        if int(item.get("size", -1)) == EXPECTED_RAW_SIZE
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "Could not identify the exact OpenCSI raw archive in Figshare: "
            + repr([(item.get("name"), item.get("size")) for item in metadata.get("files", [])])
        )

    metadata_path = WORK / "figshare_article_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    raw_path = download(
        candidates[0]["download_url"],
        WORK / "openCSI.zip",
        expected_size=EXPECTED_RAW_SIZE,
        expected_sha256=EXPECTED_RAW_SHA256,
    )
    actual_md5 = hash_file(raw_path, "md5")
    if actual_md5 != EXPECTED_RAW_MD5:
        raise RuntimeError(
            f"Raw MD5 mismatch: {actual_md5} != {EXPECTED_RAW_MD5}"
        )
    return raw_path, metadata_path


def transfer_raw_direct_from_figshare():
    print("Release assets are not available yet; using direct Figshare → Drive transfer.")
    raw_path, metadata_path = get_figshare_raw()
    drive_upload(raw_path, RAW_FOLDER_ID, "application/zip")
    drive_upload(metadata_path, CHECKSUMS_FOLDER_ID, "application/json")

    provenance = {
        "doi": "10.6084/m9.figshare.19596379.v1",
        "article_id": ARTICLE_ID,
        "raw_filename": raw_path.name,
        "raw_size_bytes": raw_path.stat().st_size,
        "raw_md5": EXPECTED_RAW_MD5,
        "raw_sha256": EXPECTED_RAW_SHA256,
        "transfer_mode": "direct_figshare_to_colab_to_google_drive",
    }
    provenance_path = WORK / "OPENCSI_RAW_PROVENANCE.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    drive_upload(provenance_path, CHECKSUMS_FOLDER_ID, "application/json")


def transfer_release_package(manifest):
    raw_parts = []
    for item in manifest["raw_parts"]:
        part_path = download(
            f"{RELEASE_BASE}/{quote(item['name'])}",
            WORK / item["name"],
            expected_size=item["size_bytes"],
            expected_sha256=item["sha256"],
        )
        raw_parts.append(part_path)

    raw_info = manifest["raw_archive"]
    raw_archive = WORK / raw_info["name"]
    reconstructing = pathlib.Path(str(raw_archive) + ".reconstructing")

    with open(reconstructing, "wb") as output:
        for part in raw_parts:
            with open(part, "rb") as source:
                for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
                    output.write(block)

    if reconstructing.stat().st_size != raw_info["size_bytes"]:
        raise RuntimeError("Reconstructed raw archive size mismatch")
    if hash_file(reconstructing, "sha256") != raw_info["sha256"]:
        raise RuntimeError("Reconstructed raw archive SHA-256 mismatch")
    if hash_file(reconstructing, "md5") != raw_info["md5"]:
        raise RuntimeError("Reconstructed raw archive MD5 mismatch")

    reconstructing.replace(raw_archive)
    drive_upload(raw_archive, RAW_FOLDER_ID, "application/zip")

    destinations = {
        "opencsi-compact-v8-measured.zip": (
            SOURCE_RESULTS_FOLDER_ID,
            "application/zip",
        ),
        "OPENCSI_FROZEN_RUNS_CHECKPOINTS.zip": (
            SOURCE_RESULTS_FOLDER_ID,
            "application/zip",
        ),
        "OPENCSI_REPOSITORY_SOURCE_38a6580.zip": (
            SOURCE_RESULTS_FOLDER_ID,
            "application/zip",
        ),
        "OPENCSI_FULL_PAPER_SRC_RESULTS_DOCKER.zip": (
            PAPER_FOLDER_ID,
            "application/zip",
        ),
        "figshare_article_metadata.json": (
            CHECKSUMS_FOLDER_ID,
            "application/json",
        ),
        "README_TRANSFER.txt": (
            CHECKSUMS_FOLDER_ID,
            "text/plain",
        ),
    }

    for item in manifest["package_assets"]:
        if item["name"] not in destinations:
            raise RuntimeError(f"Unknown release asset: {item['name']}")
        asset_path = download(
            f"{RELEASE_BASE}/{quote(item['name'])}",
            WORK / item["name"],
            expected_size=item["size_bytes"],
            expected_sha256=item["sha256"],
        )
        parent_id, mime_type = destinations[item["name"]]
        drive_upload(asset_path, parent_id, mime_type)

    manifest_path = WORK / "release_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    drive_upload(manifest_path, CHECKSUMS_FOLDER_ID, "application/json")

    checksums_path = download(
        f"{RELEASE_BASE}/SHA256SUMS.txt",
        WORK / "SHA256SUMS.txt",
    )
    drive_upload(checksums_path, CHECKSUMS_FOLDER_ID, "text/plain")


print("Destination:", f"https://drive.google.com/drive/folders/{ROOT_FOLDER_ID}")
manifest_url = f"{RELEASE_BASE}/release_manifest.json"
manifest_response = SESSION.get(manifest_url, timeout=60)

if manifest_response.status_code == 200:
    print("Stable release found; transferring the complete package.")
    transfer_release_package(manifest_response.json())
else:
    print(
        f"Stable release manifest returned HTTP {manifest_response.status_code}."
    )
    transfer_raw_direct_from_figshare()

print()
print("TRANSFER FINISHED")
print("Drive root:", f"https://drive.google.com/drive/folders/{ROOT_FOLDER_ID}")
print("Raw SHA-256:", EXPECTED_RAW_SHA256)
