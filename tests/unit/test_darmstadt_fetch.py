"""Checks for the Darmstadt range fetcher's ZIP directory parser."""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path


def test_zip64_target_size_and_offset(monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts/darmstadt/fetch_biomechanics.py"
    spec = importlib.util.spec_from_file_location("darmstadt_fetch_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    large_size = 5_200_000_000
    large_compressed = 4_500_000_000
    large_offset = 5_000_000_000
    names = [
        *(f"Preprocessed/EMG/EMG{subject}.mat" for subject in range(1, 13)),
        *(f"Preprocessed/Forces/Forces{subject}.mat" for subject in range(1, 13)),
    ]
    directory = bytearray()
    for index, name in enumerate(names):
        filename = name.encode()
        if index == 0:
            size = compressed_size = local_offset = 0xFFFFFFFF
            extra = struct.pack("<HHQQQ", 0x0001, 24, large_size, large_compressed, large_offset)
        else:
            size = compressed_size = 1
            local_offset = index
            extra = b""
        directory.extend(
            struct.pack(
                "<4s6H3I5H2I",
                b"PK\x01\x02", 45, 45, 0, 8, 0, 0, 0,
                compressed_size, size, len(filename), len(extra), 0, 0, 0, 0, local_offset,
            )
        )
        directory.extend(filename)
        directory.extend(extra)

    entries = module._central_entries(bytes(directory))
    first = next(entry for entry in entries if entry.name == names[0])
    assert (first.size, first.compressed_size, first.local_offset) == (
        large_size,
        large_compressed,
        large_offset,
    )
