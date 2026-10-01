#!/usr/bin/env python3
"""Fetch and verify the small Vielemeyer calculated-NPZ subject archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
import zipfile
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the official RefNN.zip calculated-data archives listed in the "
            "Vielemeyer Figshare metadata and verify their MD5 checksums."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("data/Vielemeyer-Ramp-Walking"),
        help="Vielemeyer dataset root (default: data/Vielemeyer-Ramp-Walking)",
    )
    return parser


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = _parser().parse_args()
    root = args.root.resolve()
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    entries = [entry for entry in metadata["files"] if re.fullmatch(r"Ref\d{2}\.zip", entry["name"])]
    if len(entries) != 11:
        raise ValueError(f"expected 11 calculated-data archives in {metadata_path}, found {len(entries)}")
    root.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for entry in entries:
        target = root / entry["name"]
        expected_md5 = entry["computed_md5"]
        if target.is_file() and target.stat().st_size == entry["size"] and _md5(target) == expected_md5:
            print(f"verified {target.name}")
            continue
        temporary = target.with_name(f".{target.name}.download")
        print(f"downloading {target.name} ({entry['size'] / 1_000_000:.1f} MB)")
        urllib.request.urlretrieve(entry["download_url"], temporary)
        if temporary.stat().st_size != entry["size"] or _md5(temporary) != expected_md5:
            raise ValueError(f"download verification failed for {entry['name']}")
        with zipfile.ZipFile(temporary) as archive:
            bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"ZIP CRC failed for {entry['name']} member {bad_member}")
        temporary.replace(target)
        downloaded += 1
    print(f"Vielemeyer calculated archives ready: {len(entries)} ({downloaded} downloaded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
