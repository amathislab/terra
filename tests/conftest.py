"""Shared test setup and optional distribution checks."""

import shutil
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run distribution build and installation tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="use --runslow to run distribution checks")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def release_source(tmp_path_factory):
    """Copy versioned source inputs without a checkout or local build artifacts."""
    repo = Path(__file__).resolve().parents[1]
    source = tmp_path_factory.mktemp("release-source")
    for filename in ("pyproject.toml", "MANIFEST.in", "uv.lock", "README.md", "LICENSE", ".python-version"):
        shutil.copy2(repo / filename, source / filename)
    for directory in ("src", "scripts", "docs", "tests"):
        shutil.copytree(
            repo / directory, source / directory, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info")
        )
    return source
