"""
Locate the repository's ComfyUI node packages so the offline scripts can import
their maths modules.

The packages ship in this repository under comfyui/custom_nodes/, and that copy
is the one the scripts use: resolving relative to this file works from any clone
on any platform, whereas an absolute path into one machine's ComfyUI install
works only on that machine.

Set L3DGEN_NODES to point somewhere else — useful when the installed copy has
diverged from the repository and you want to test what ComfyUI is actually
running.
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_NODES = os.path.join(REPO, "comfyui", "custom_nodes")


def nodes_root():
    """Directory holding the comfyui_* package folders."""
    return os.environ.get("L3DGEN_NODES") or DEFAULT_NODES


def add(package):
    """Put one node package on sys.path and return its directory."""
    path = os.path.join(nodes_root(), package)
    if not os.path.isdir(path):
        raise SystemExit(
            "Could not find '%s' in %s.\n"
            "Run this from a clone of the repository, or set L3DGEN_NODES to the "
            "directory containing the comfyui_* packages." % (package, nodes_root()))
    if path not in sys.path:
        sys.path.insert(0, path)
    return path
