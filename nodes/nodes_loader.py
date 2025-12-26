"""Model loading nodes for WaLa."""
import torch

import comfy.model_management as mm

from .utils import logger

# WaLa model variants
WALA_MODELS = [
    'ADSKAILab/WaLa-SV-1B',      # Single-view
    'ADSKAILab/WaLa-RGB4-1B',    # Multi-view (4 RGB)
    'ADSKAILab/WaLa-DM1-1B',     # Single depth map
    'ADSKAILab/WaLa-DM4-1B',     # 4 depth maps
    'ADSKAILab/WaLa-DM6-1B',     # 6 depth maps
    'ADSKAILab/WaLa-PC-1B',      # Pointcloud
]

# MVDream model variants
MVDREAM_MODELS = [
    'ADSKAILab/WaLa-MVDream-DM6',   # Text to 6 depth maps
    'ADSKAILab/WaLa-MVDream-RGB4',  # Text to 4 multi-view RGB
]

# Model caches - keeps loaded models in memory when keep_model_loaded is enabled
_wala_model_cache = {}
_mvdream_model_cache = {}


class LoadWaLaModel:
    """Load WaLa model for 3D generation."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (WALA_MODELS, {"default": 'ADSKAILab/WaLa-SV-1B'}),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model in memory between runs for faster subsequent generations"
                }),
            },
        }

    RETURN_TYPES = ("WALA_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loadmodel"
    CATEGORY = "WaLa"
    DESCRIPTION = """
Load WaLa model for 3D generation from various inputs.

Available models:
- WaLa-SV-1B: Single-view image to 3D
- WaLa-RGB4-1B: 4 multi-view RGB images to 3D
- WaLa-DM1-1B: Single depth map to 3D
- WaLa-DM4-1B: 4 depth maps to 3D
- WaLa-DM6-1B: 6 depth maps to 3D
- WaLa-PC-1B: Pointcloud to 3D

Models are downloaded from HuggingFace on first use.
"""

    def loadmodel(self, model_name='ADSKAILab/WaLa-SV-1B', keep_model_loaded=True):
        device = mm.get_torch_device()
        cache_key = f"{model_name}_{device}"

        # Check cache first
        if keep_model_loaded and cache_key in _wala_model_cache:
            logger.info(f"Using cached WaLa model: {model_name}")
            return (_wala_model_cache[cache_key],)

        logger.info(f"Loading WaLa model: {model_name}")

        from ..wala.model_utils import Model

        model = Model.from_pretrained(pretrained_model_name_or_path=model_name)

        logger.info(f"WaLa model loaded successfully on {model.device}")

        wala_model = {
            "model": model,
            "model_name": model_name,
            "device": model.device,
        }

        # Cache if requested
        if keep_model_loaded:
            _wala_model_cache[cache_key] = wala_model
            logger.info(f"WaLa model cached for future use")

        return (wala_model,)


class LoadWaLaMVDream:
    """Load WaLa MVDream model for text-to-3D."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (MVDREAM_MODELS, {"default": 'ADSKAILab/WaLa-MVDream-DM6'}),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep model in memory between runs for faster subsequent generations"
                }),
            },
        }

    RETURN_TYPES = ("WALA_MVDREAM",)
    RETURN_NAMES = ("mvdream_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "WaLa"
    DESCRIPTION = """
Load WaLa MVDream model for text-to-3D generation.

Available models:
- WaLa-MVDream-DM6: Text to 6 depth maps
- WaLa-MVDream-RGB4: Text to 4 multi-view RGB images

The generated depth maps or RGB images can then be used with
the corresponding WaLa model (DM6 or RGB4) to create 3D meshes.

Models are downloaded from HuggingFace on first use.
"""

    def loadmodel(self, model_name='ADSKAILab/WaLa-MVDream-DM6', keep_model_loaded=True):
        device = mm.get_torch_device()
        cache_key = f"{model_name}_{device}"

        # Check cache first
        if keep_model_loaded and cache_key in _mvdream_model_cache:
            logger.info(f"Using cached WaLa MVDream model: {model_name}")
            return (_mvdream_model_cache[cache_key],)

        logger.info(f"Loading WaLa MVDream model: {model_name}")

        from ..wala.mvdream_utils import load_mvdream_model

        model = load_mvdream_model(
            pretrained_model_name_or_path=model_name,
            device=device
        )

        logger.info(f"WaLa MVDream model loaded successfully")

        mvdream_model = {
            "model": model,
            "model_name": model_name,
            "device": device,
        }

        # Cache if requested
        if keep_model_loaded:
            _mvdream_model_cache[cache_key] = mvdream_model
            logger.info(f"WaLa MVDream model cached for future use")

        return (mvdream_model,)


NODE_CLASS_MAPPINGS = {
    "LoadWaLaModel": LoadWaLaModel,
    "LoadWaLaMVDream": LoadWaLaMVDream,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadWaLaModel": "Load WaLa Model",
    "LoadWaLaMVDream": "Load WaLa MVDream",
}
