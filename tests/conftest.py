"""Shared test setup and optional distribution checks."""

import shutil
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run distribution and real solver integration tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="use --runslow to run distribution and integration checks")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def test_revision(monkeypatch):
    """Use a supplied revision so tests also run from ZIP and source distributions."""
    monkeypatch.setenv("TERRA_GIT_COMMIT", "1234567890abcdef1234567890abcdef12345678")


@pytest.fixture(scope="session")
def release_source(tmp_path_factory):
    """Copy versioned source inputs without a checkout or local build artifacts."""
    repo = Path(__file__).resolve().parents[1]
    source = tmp_path_factory.mktemp("release-source")
    for filename in ("pyproject.toml", "MANIFEST.in", "uv.lock", "README.md", "LICENSE", "NOTICE", ".python-version"):
        shutil.copy2(repo / filename, source / filename)
    for directory in ("src", "scripts", "docs", "tests"):
        shutil.copytree(
            repo / directory, source / directory, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info")
        )
    return source
