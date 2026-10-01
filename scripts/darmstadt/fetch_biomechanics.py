#!/usr/bin/env python3
"""Fetch only Darmstadt's trial-level EMG and force MAT files from Preprocessed.zip."""

from __future__ import annotations

import argparse
import binascii
import os
import struct
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path

from terra.paths import StorageRoots

ARCHIVE_URL = (
    "https://tudatalib.ulb.tu-darmstadt.de/server/api/core/bitstreams/"
    "ffc932af-a7c6-43a1-aa4f-66c74b2ed531/content"
)
ARCHIVE_SIZE = 4_520_656_121
TAIL_SIZE = 2 * 1024 * 1024
RANGE_CHUNK_SIZE = 32 * 1024 * 1024


@dataclass(frozen=True)
class ZipEntry:
    name: str
    method: int
    crc32: int
    compressed_size: int
    size: int
    local_offset: int


def _single_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request) as response:
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
            raise RuntimeError(
                f"Server ignored exact byte range {start}-{end}: "
                f"status={response.status}, Content-Range={content_range!r}"
            )
        payload = response.read()
    expected = end - start + 1
    if len(payload) != expected:
        raise RuntimeError(f"Range {start}-{end} returned {len(payload)} bytes, expected {expected}")
    return payload


def _range(url: str, start: int, end: int) -> bytes:
    chunks = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(end, chunk_start + RANGE_CHUNK_SIZE - 1)
        chunks.append(_single_range(url, chunk_start, chunk_end))
        chunk_start = chunk_end + 1
    return b"".join(chunks)


def _zip64_values(extra: bytes, size: int, compressed_size: int, local_offset: int) -> tuple[int, int, int]:
    """Read values replaced by ZIP64 sentinels in a central-directory entry."""
    if 0xFFFFFFFF not in (size, compressed_size, local_offset):
        return size, compressed_size, local_offset
    position = 0
    while position + 4 <= len(extra):
        kind, length = struct.unpack_from("<HH", extra, position)
        position += 4
        end = position + length
        if end > len(extra):
            raise RuntimeError("Malformed ZIP extra field")
        if kind == 0x0001:
            values = [size, compressed_size, local_offset]
            for index, value in enumerate(values):
                if value == 0xFFFFFFFF:
                    if position + 8 > end:
                        raise RuntimeError("ZIP64 extra field is missing a required size or offset")
                    values[index] = struct.unpack_from("<Q", extra, position)[0]
                    position += 8
            return values[0], values[1], values[2]
        position = end
    raise RuntimeError("ZIP64 extra field is missing for a saturated size or offset")


def _central_entries(tail: bytes) -> list[ZipEntry]:
    entries = []
    position = 0
    while True:
        position = tail.find(b"PK\x01\x02", position)
        if position < 0:
            break
        if position + 46 > len(tail):
            break
        fields = struct.unpack_from("<4s6H3I5H2I", tail, position)
        (
            _signature,
            _version_made,
            _version_needed,
            _flags,
            method,
            _mtime,
            _mdate,
            crc32,
            compressed_size,
            size,
            filename_length,
            extra_length,
            comment_length,
            _disk,
            _internal,
            _external,
            local_offset,
        ) = fields
        name_start = position + 46
        name = tail[name_start : name_start + filename_length].decode("utf-8")
        if name.startswith(("Preprocessed/EMG/EMG", "Preprocessed/Forces/Forces")) and name.endswith(".mat"):
            extra = tail[name_start + filename_length : name_start + filename_length + extra_length]
            size, compressed_size, local_offset = _zip64_values(extra, size, compressed_size, local_offset)
            entries.append(
                ZipEntry(name, method, crc32, compressed_size, size, local_offset)
            )
        position = name_start + filename_length + extra_length + comment_length
    expected = {
        *(f"Preprocessed/EMG/EMG{subject}.mat" for subject in range(1, 13)),
        *(f"Preprocessed/Forces/Forces{subject}.mat" for subject in range(1, 13)),
    }
    actual = {entry.name for entry in entries}
    if actual != expected:
        raise RuntimeError(f"ZIP directory is missing expected entries: {sorted(expected - actual)}")
    return sorted(entries, key=lambda entry: entry.local_offset)


def _entry_data(url: str, entry: ZipEntry) -> bytes:
    header = _range(url, entry.local_offset, entry.local_offset + 29)
    signature, _version, flags, method, *_rest, filename_length, extra_length = struct.unpack(
        "<4s5H3I2H", header
    )
    if signature != b"PK\x03\x04" or flags & 0x08 or method != entry.method:
        raise RuntimeError(f"Unsupported local ZIP header for {entry.name}")
    compressed_start = entry.local_offset + 30 + filename_length + extra_length
    compressed = _range(
        url,
        compressed_start,
        compressed_start + entry.compressed_size - 1,
    )
    if method == 0:
        payload = compressed
    elif method == 8:
        payload = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise RuntimeError(f"Unsupported ZIP method {method} for {entry.name}")
    if len(payload) != entry.size:
        raise RuntimeError(f"{entry.name} has size {len(payload)}, expected {entry.size}")
    checksum = binascii.crc32(payload) & 0xFFFFFFFF
    if checksum != entry.crc32:
        raise RuntimeError(f"{entry.name} CRC {checksum:08x}, expected {entry.crc32:08x}")
    return payload


def _valid_existing(path: Path, entry: ZipEntry) -> bool:
    if not path.is_file() or path.stat().st_size != entry.size:
        return False
    checksum = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            checksum = binascii.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF == entry.crc32


def fetch(output_root: Path, *, url: str = ARCHIVE_URL) -> None:
    tail = _range(url, ARCHIVE_SIZE - TAIL_SIZE, ARCHIVE_SIZE - 1)
    entries = _central_entries(tail)
    for index, entry in enumerate(entries, start=1):
        relative = Path(entry.name).relative_to("Preprocessed")
        output = output_root / "Preprocessed" / relative
        if _valid_existing(output, entry):
            print(f"[{index}/24] existing {relative}", flush=True)
            continue
        print(
            f"[{index}/24] downloading {relative} ({entry.compressed_size / 1024**2:.1f} MiB)",
            flush=True,
        )
        payload = _entry_data(url, entry)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            handle.write(payload)
        temporary.replace(output)


def main() -> None:
    roots = StorageRoots.from_environment(Path.cwd())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=roots.data_root / "Darmstadt-Stair-Ambulation",
    )
    parser.add_argument("--url", default=ARCHIVE_URL)
    args = parser.parse_args()
    fetch(args.output_root.resolve(), url=args.url)


if __name__ == "__main__":
    main()
