"""
Path helpers for SerdesProjectPython.

Layout::

    SerdesProjectPython/
      path_setup.py          <-- this file
      notebooks/
      serdes_building_blocks/

Parent directory (workspace root, e.g. CursorProjPython) may contain
Verilog_testing_v2/, Serdes_v1_ffe_channel_ctle/, etc.
"""

from __future__ import annotations

import sys
from pathlib import Path


def serdes_project_root() -> Path:
    """
    Directory that contains both ``notebooks/`` and ``serdes_building_blocks/``.

    Resolves whether your Jupyter **working directory** is ``SerdesProjectPython``,
    ``SerdesProjectPython/notebooks``, or the **workspace root** that *contains*
    ``SerdesProjectPython/`` (common in VS Code/Cursor).
    """
    p = Path.cwd().resolve()
    candidates: list[Path] = [p]
    if p.name == "notebooks":
        candidates.append(p.parent)
        # e.g. .../CursorProjPython/notebooks → sibling SerdesProjectPython
        sp_sibling = p.parent / "SerdesProjectPython"
        if (sp_sibling / "serdes_building_blocks").is_dir() and (
            sp_sibling / "notebooks"
        ).is_dir():
            candidates.append(sp_sibling.resolve())

    # e.g. cwd is workspace root .../CursorProjPython → nested SerdesProjectPython/
    nested = p / "SerdesProjectPython"
    if (nested / "serdes_building_blocks").is_dir() and (nested / "notebooks").is_dir():
        candidates.append(nested.resolve())

    for c in candidates:
        bb = c / "serdes_building_blocks"
        nb = c / "notebooks"
        if bb.is_dir() and nb.is_dir():
            return c.resolve()

    for ancestor in p.parents:
        bb = ancestor / "serdes_building_blocks"
        nb = ancestor / "notebooks"
        if bb.is_dir() and nb.is_dir():
            return ancestor.resolve()

    raise RuntimeError(
        "Could not find SerdesProjectPython (need a directory containing "
        "both notebooks/ and serdes_building_blocks/). "
        "Try: cd SerdesProjectPython from the terminal, or open the folder "
        "SerdesProjectPython in Cursor so cwd matches."
    )


def workspace_root() -> Path:
    """Parent of SerdesProjectPython (e.g. CursorProjPython)."""
    return serdes_project_root().parent


def add_serdes_project_to_path() -> None:
    """Insert SerdesProjectPython so ``import path_setup`` works from scripts."""
    root = serdes_project_root()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


def add_building_blocks() -> None:
    """Prefer ``serdes_building_blocks`` for imports (``serdes_channel``, etc.)."""
    bb = serdes_project_root() / "serdes_building_blocks"
    s = str(bb)
    if s not in sys.path:
        sys.path.insert(0, s)


def add_verilog_testing_v2() -> None:
    """Import scripts from sibling ``Verilog_testing_v2`` under workspace root."""
    v = workspace_root() / "Verilog_testing_v2"
    s = str(v)
    if s not in sys.path:
        sys.path.insert(0, s)


def add_serdes_v1_ffe_channel_ctle() -> None:
    """Optional: legacy experiments under workspace ``Serdes_v1_ffe_channel_ctle``."""
    v = workspace_root() / "Serdes_v1_ffe_channel_ctle"
    s = str(v)
    if s not in sys.path:
        sys.path.insert(0, s)
