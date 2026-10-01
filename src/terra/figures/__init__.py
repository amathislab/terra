"""Manifest-driven publication figures built from frozen TERRA artifacts.

Rendering backends are imported lazily because MuJoCo selects its headless GL backend at
module-import time.  The manifest and layout modules remain lightweight and testable.
"""

__all__: list[str] = []
