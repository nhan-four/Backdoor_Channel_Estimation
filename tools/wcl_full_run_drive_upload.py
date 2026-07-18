from google.colab import auth, files
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

import hashlib
import json
import pathlib
import time

# Exact frozen package produced by RUN_ALL_WCL_EXTENSION.sh on 2026-07-18.
EXPECTED_NAME = "WCL_EXTENSION_FULL_RUN_COMPLETE_20260718.zip"
EXPECTED_SIZE = 155_612_278
EXPECTED_SHA256 = "ee64b6ee9348e31015177fcd767b035757c27b9bf6c9f22b8607704b26701ee4"
EXPECTED_MD5 = "a9305ce0f582d6435ff4dd2fe890f3a9"

# Destination created for this full run.
TARGET_FOLDER_ID = "14oyYPdDJ3NDljj5TVK-gM8CbGnD7Lj7f"  # full_run_source_and_results
CHECKSUMS_FOLDER_ID = "1FxxPxaUlq8ThJKO3dPsAMj040Mb_HRbn"


def hash_file(path: pathlib.Path, algorithm: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_local(path: pathlib.Path) -> dict:
    if path.name != EXPECTED_NAME:
        raise RuntimeError(f"Wrong filename: {path.name!r}; expected {EXPECTED_NAME!r}")
    size = path.stat().st_size
    sha256 = hash_file(path, "sha256")
    md5 = hash_file(path, "md5")
    if size != EXPECTED_SIZE:
        raise RuntimeError(f"Size mismatch: {size} != {EXPECTED_SIZE}")
    if sha256 != EXPECTED_SHA256:
        raise RuntimeError(f"SHA-256 mismatch: {sha256} != {EXPECTED_SHA256}")
    if md5 != EXPECTED_MD5:
        raise RuntimeError(f"MD5 mismatch: {md5} != {EXPECTED_MD5}")
    return {"name": path.name, "size": size, "sha256": sha256, "md5": md5}


def find_files(drive, parent_id: str, name: str):
    safe_name = name.replace("'", "\\'")
    query = f"'{parent_id}' in parents and name = '{safe_name}' and trashed = false"
    response = drive.files().list(
        q=query,
        fields="files(id,name,size,md5Checksum,webViewLink,createdTime,modifiedTime)",
        pageSize=100,
    ).execute()
    return response.get("files", [])


def resumable_create_or_replace(drive, path: pathlib.Path, parent_id: str, mime_type: str):
    media = MediaFileUpload(
        str(path),
        mimetype=mime_type,
        resumable=True,
        chunksize=32 * 1024 * 1024,
    )
    existing = find_files(drive, parent_id, path.name)

    if existing:
        # The first item is the placeholder created earlier; replace bytes in place
        # so the Drive URL remains stable. Any extra same-name copies are trashed.
        target = existing[0]
        request = drive.files().update(
            fileId=target["id"],
            media_body=media,
            fields="id,name,size,md5Checksum,webViewLink,createdTime,modifiedTime",
        )
        operation = "REPLACED"
        for duplicate in existing[1:]:
            drive.files().update(fileId=duplicate["id"], body={"trashed": True}).execute()
    else:
        request = drive.files().create(
            body={"name": path.name, "parents": [parent_id]},
            media_body=media,
            fields="id,name,size,md5Checksum,webViewLink,createdTime,modifiedTime",
        )
        operation = "CREATED"

    result = None
    while result is None:
        status, result = request.next_chunk(num_retries=10)
        if status:
            print(f"  Drive upload: {status.progress() * 100:.1f}%")
    print(operation, result.get("webViewLink", result["id"]))
    return result


def write_sidecars(local_info: dict):
    checksum = pathlib.Path(EXPECTED_NAME + ".sha256")
    checksum.write_text(f"{EXPECTED_SHA256}  {EXPECTED_NAME}\n", encoding="utf-8")

    manifest = pathlib.Path("WCL_EXTENSION_FULL_RUN_DRIVE_MANIFEST_20260718.json")
    manifest.write_text(
        json.dumps(
            {
                "package": local_info,
                "scientific_status": "integrity_attribution_audit; detector_negative; conditional_repair_positive",
                "expected_detection_rows": 288,
                "expected_repair_rows": 72,
                "expected_receiver_raw_rows": 2016,
                "target_folder_id": TARGET_FOLDER_ID,
                "transfer_method": "user-authorized_colab_resumable_upload",
                "generated_unix_time": int(time.time()),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return checksum, manifest


print("Authenticate the Google account that owns the OpenCSI Drive folder.")
auth.authenticate_user()
drive = build("drive", "v3")

print()
print(f"Select exactly: {EXPECTED_NAME}")
uploaded = files.upload()
if EXPECTED_NAME not in uploaded:
    raise RuntimeError(
        f"Required file not selected. Uploaded names: {sorted(uploaded)}"
    )

package_path = pathlib.Path(EXPECTED_NAME)
local_info = verify_local(package_path)
print("LOCAL VERIFICATION PASSED")
print(json.dumps(local_info, indent=2))

remote = resumable_create_or_replace(
    drive, package_path, TARGET_FOLDER_ID, "application/zip"
)
if int(remote.get("size", -1)) != EXPECTED_SIZE:
    raise RuntimeError(f"Drive size mismatch: {remote.get('size')} != {EXPECTED_SIZE}")
if remote.get("md5Checksum", "").lower() != EXPECTED_MD5:
    raise RuntimeError(
        f"Drive MD5 mismatch: {remote.get('md5Checksum')} != {EXPECTED_MD5}"
    )

checksum_path, manifest_path = write_sidecars(local_info)
for sidecar in (checksum_path, manifest_path):
    parent = CHECKSUMS_FOLDER_ID
    mime = "application/json" if sidecar.suffix == ".json" else "text/plain"
    result = resumable_create_or_replace(drive, sidecar, parent, mime)
    if int(result.get("size", -1)) != sidecar.stat().st_size:
        raise RuntimeError(f"Drive sidecar size mismatch for {sidecar.name}")

# Final independent metadata readback.
final_meta = drive.files().get(
    fileId=remote["id"],
    fields="id,name,size,md5Checksum,webViewLink,parents,modifiedTime",
).execute()
print()
print("TRANSFER FINISHED")
print(json.dumps(final_meta, indent=2))
print("Expected SHA-256:", EXPECTED_SHA256)
print("Drive folder:", f"https://drive.google.com/drive/folders/{TARGET_FOLDER_ID}")
