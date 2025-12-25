"""
ComfyUI-WaLa: WaLa 3D Generation nodes for ComfyUI

WaLa is a billion-parameter 3D generative model that supports multiple input modalities:
- Single-view image to 3D
- Multi-view images to 3D
- Depth maps to 3D
- Pointcloud to 3D
- Text to 3D (via MVDream)
"""

import sys
import os
import traceback

# Track initialization status
INIT_SUCCESS = False
INIT_ERRORS = []

# Web directory for JavaScript extensions (if needed in future)
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

# Only run initialization and imports when loaded by ComfyUI, not during pytest
if 'PYTEST_CURRENT_TEST' not in os.environ:
    print("[ComfyUI-WaLa] Initializing custom node...")

    # Add wala directory to path for imports
    wala_path = os.path.join(os.path.dirname(__file__), "wala")
    if wala_path not in sys.path:
        sys.path.insert(0, wala_path)

    try:
        from .nodes import (
            NODE_CLASS_MAPPINGS,
            NODE_DISPLAY_NAME_MAPPINGS,
        )
        print("[ComfyUI-WaLa] [OK] Node classes imported successfully")
        INIT_SUCCESS = True
    except Exception as e:
        error_msg = f"Failed to import node classes: {str(e)}"
        INIT_ERRORS.append(error_msg)
        print(f"[ComfyUI-WaLa] [WARNING] {error_msg}")
        print(f"[ComfyUI-WaLa] Traceback:\n{traceback.format_exc()}")

        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}

    if INIT_SUCCESS:
        print("[ComfyUI-WaLa] [OK] Loaded successfully!")
        print("[ComfyUI-WaLa] Available nodes:")
        for node_name in NODE_CLASS_MAPPINGS:
            print(f"  - {node_name}")
    else:
        print(f"[ComfyUI-WaLa] [ERROR] Failed to load ({len(INIT_ERRORS)} error(s)):")
        for error in INIT_ERRORS:
            print(f"  - {error}")
        print("[ComfyUI-WaLa] Please check the errors above and your installation.")

else:
    print("[ComfyUI-WaLa] Running in pytest mode - skipping initialization")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
