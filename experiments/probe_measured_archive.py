#!/usr/bin/env python3
"""Create a provenance-only inventory of a downloaded measured-channel archive.

This utility never reports attack/estimation performance.  It only records the
archive checksum, ZIP membership, and MATLAB variable metadata needed to freeze
a scientifically reproducible ingestion manifest before model training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable


@dataclass(frozen=True)
class MemberRecord:
    container: str
    path: str
    size_bytes: int
    compressed_bytes: int
    crc32: str
    kind: str


@dataclass(frozen=True)
class MatVariableRecord:
    container: str
    path: str
    variable: str
    shape: str
    dtype_or_class: str
    backend: str


def hash_file(path: Path, algorithm: str, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(path: str) -> str:
    suffix = PurePosixPath(path).suffix.lower()
    if suffix == ".mat":
        return "mat"
    if suffix == ".zip":
        return "zip"
    if suffix in {".csv", ".txt", ".md", ".json", ".m", ".py"}:
        return "text_or_code"
    if suffix in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
        return "image"
    return "other"


def safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def inspect_mat_bytes(
    raw: bytes,
    *,
    container: str,
    path: str,
) -> list[MatVariableRecord]:
    records: list[MatVariableRecord] = []

    # MATLAB v5/v7 (non-HDF5) files.
    try:
        import scipy.io  # type: ignore

        for variable, shape, matlab_class in scipy.io.whosmat(io.BytesIO(raw)):
            records.append(
                MatVariableRecord(
                    container=container,
                    path=path,
                    variable=str(variable),
                    shape="x".join(str(value) for value in shape),
                    dtype_or_class=str(matlab_class),
                    backend="scipy.io.whosmat",
                )
            )
        if records:
            return records
    except Exception:
        pass

    # MATLAB v7.3 files use HDF5.  Write to a temporary file because h5py is
    # more reliable with a filesystem path across versions.
    try:
        import h5py  # type: ignore

        with tempfile.NamedTemporaryFile(suffix=".mat") as tmp:
            tmp.write(raw)
            tmp.flush()
            with h5py.File(tmp.name, "r") as handle:
                def visitor(name: str, obj: object) -> None:
                    if isinstance(obj, h5py.Dataset):
                        records.append(
                            MatVariableRecord(
                                container=container,
                                path=path,
                                variable=name,
                                shape="x".join(str(value) for value in obj.shape),
                                dtype_or_class=str(obj.dtype),
                                backend="h5py",
                            )
                        )

                handle.visititems(visitor)
    except Exception as exc:
        records.append(
            MatVariableRecord(
                container=container,
                path=path,
                variable="<unreadable>",
                shape="",
                dtype_or_class=f"{type(exc).__name__}: {exc}",
                backend="failed",
            )
        )
    return records


def inspect_zip(
    source: Path | BinaryIO,
    *,
    container_name: str,
    member_records: list[MemberRecord],
    mat_records: list[MatVariableRecord],
    max_nested_depth: int,
    depth: int = 0,
) -> None:
    with zipfile.ZipFile(source) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"CRC failure in {container_name}: {bad_member}")

        for info in archive.infolist():
            if info.is_dir():
                continue
            if not safe_member_name(info.filename):
                raise RuntimeError(
                    f"Unsafe archive member in {container_name}: {info.filename}"
                )
            kind = classify(info.filename)
            member_records.append(
                MemberRecord(
                    container=container_name,
                    path=info.filename,
                    size_bytes=int(info.file_size),
                    compressed_bytes=int(info.compress_size),
                    crc32=f"{info.CRC:08x}",
                    kind=kind,
                )
            )

            if kind == "mat":
                with archive.open(info, "r") as handle:
                    raw = handle.read()
                mat_records.extend(
                    inspect_mat_bytes(raw, container=container_name, path=info.filename)
                )
            elif kind == "zip" and depth < max_nested_depth:
                with archive.open(info, "r") as handle:
                    nested = io.BytesIO(handle.read())
                inspect_zip(
                    nested,
                    container_name=f"{container_name}!/{info.filename}",
                    member_records=member_records,
                    mat_records=mat_records,
                    max_nested_depth=max_nested_depth,
                    depth=depth + 1,
                )


def write_csv(path: Path, rows: Iterable[object], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path)
    parser.add_argument("--max-nested-depth", type=int, default=3)
    args = parser.parse_args()

    archive_path = args.archive.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    if not zipfile.is_zipfile(archive_path):
        raise RuntimeError(f"Not a valid ZIP archive: {archive_path}")

    member_records: list[MemberRecord] = []
    mat_records: list[MatVariableRecord] = []
    inspect_zip(
        archive_path,
        container_name=archive_path.name,
        member_records=member_records,
        mat_records=mat_records,
        max_nested_depth=args.max_nested_depth,
    )

    md5 = hash_file(archive_path, "md5")
    sha256 = hash_file(archive_path, "sha256")
    source_metadata: dict[str, object] = {}
    if args.source_metadata and args.source_metadata.is_file():
        source_metadata = json.loads(args.source_metadata.read_text(encoding="utf-8"))

    manifest = {
        "schema_version": 1,
        "purpose": "provenance_and_structure_only",
        "paper_eligible": False,
        "archive_name": archive_path.name,
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_md5": md5,
        "archive_sha256": sha256,
        "member_count": len(member_records),
        "mat_file_count": sum(row.kind == "mat" for row in member_records),
        "nested_zip_count": sum(row.kind == "zip" for row in member_records),
        "mat_variable_count": len(mat_records),
        "source_metadata": source_metadata,
    }
    (output_dir / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(
        output_dir / "archive_members.csv",
        member_records,
        ["container", "path", "size_bytes", "compressed_bytes", "crc32", "kind"],
    )
    write_csv(
        output_dir / "mat_variables.csv",
        mat_records,
        ["container", "path", "variable", "shape", "dtype_or_class", "backend"],
    )

    report_lines = [
        "# Measured-channel archive probe",
        "",
        "> **PROVENANCE/STRUCTURE ONLY — NOT PAPER-ELIGIBLE EXPERIMENTAL EVIDENCE.**",
        "",
        f"- Archive: `{archive_path.name}`",
        f"- Size: `{archive_path.stat().st_size}` bytes",
        f"- MD5: `{md5}`",
        f"- SHA-256: `{sha256}`",
        f"- ZIP members: `{len(member_records)}`",
        f"- MATLAB files: `{manifest['mat_file_count']}`",
        f"- Nested ZIP files: `{manifest['nested_zip_count']}`",
        f"- MATLAB variables/datasets inventoried: `{len(mat_records)}`",
        "",
        "The raw archive is intentionally excluded from the uploaded workflow artifact.",
    ]
    (output_dir / "probe_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
