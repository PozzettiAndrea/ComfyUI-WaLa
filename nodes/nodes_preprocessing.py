"""Preprocessing nodes for WaLa - background removal and image preparation."""
import gc
import torch
import numpy as np
from PIL import Image

import comfy.model_management as mm

from .utils import logger, tensor_to_pil, pil_to_tensor


# Model cache - keeps loaded BiRefNet models in memory
_birefnet_cache = {}


def get_birefnet_model(model_variant: str, device: torch.device):
    """Load and cache BiRefNet model."""
    cache_key = f"{model_variant}_{device}"

    if cache_key not in _birefnet_cache:
        from transformers import AutoModelForImageSegmentation

        repo_id = f"ZhengPeng7/{model_variant}"
        logger.info(f"Loading BiRefNet model from {repo_id}...")

        model = AutoModelForImageSegmentation.from_pretrained(
            repo_id,
            trust_remote_code=True
        )
        model = model.to(device).eval()

        _birefnet_cache[cache_key] = model
        logger.info(f"BiRefNet model loaded and cached")

    return _birefnet_cache[cache_key]


class WaLaRemoveBackground:
    """Remove background from image using BiRefNet."""

    MODEL_VARIANTS = [
        "BiRefNet",           # General use
        "BiRefNet-portrait",  # Portraits
        "BiRefNet-matting",   # Matting
        "BiRefNet-HR",        # High resolution (2048x2048)
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_variant": (s.MODEL_VARIANTS, {"default": "BiRefNet"}),
            },
            "optional": {
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Mask threshold (0.5 = balanced, lower = more foreground)"
                }),
                "refine_foreground": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply mask to create clean RGBA output"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "remove_background"
    CATEGORY = "WaLa/Preprocessing"

    DESCRIPTION = """
Remove background from image using BiRefNet.

Model variants:
- BiRefNet: General purpose (recommended)
- BiRefNet-portrait: Optimized for portraits
- BiRefNet-matting: For matting/compositing
- BiRefNet-HR: High resolution (2048x2048)

Parameters:
- threshold: Confidence cutoff for mask (default 0.5)
- refine_foreground: Apply mask to image for clean edges

Returns:
- image: RGBA image with transparent background
- mask: Binary foreground mask (1=foreground, 0=background)

Models are downloaded from HuggingFace on first use.
"""

    def remove_background(self, image, model_variant="BiRefNet", threshold=0.5, refine_foreground=True):
        from torchvision import transforms

        device = mm.get_torch_device()
        model = get_birefnet_model(model_variant, device)

        # Process each image in batch
        batch_size = image.shape[0]
        output_images = []
        output_masks = []

        for i in range(batch_size):
            # Get single image [H, W, C]
            img_tensor = image[i]
            pil_image = tensor_to_pil(img_tensor.unsqueeze(0))
            original_size = pil_image.size  # (W, H)

            logger.info(f"Processing image {i+1}/{batch_size}, size: {original_size}")

            # Prepare input for BiRefNet
            # BiRefNet expects normalized tensor
            input_size = (1024, 1024)  # BiRefNet input resolution

            transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            input_tensor = transform(pil_image).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                outputs = model(input_tensor)
                # BiRefNet returns list of outputs, take the last one (finest)
                if isinstance(outputs, (list, tuple)):
                    mask_logits = outputs[-1]
                else:
                    mask_logits = outputs

            # Sigmoid to get probabilities
            mask_prob = torch.sigmoid(mask_logits)

            # Resize mask back to original size
            mask_resized = torch.nn.functional.interpolate(
                mask_prob,
                size=(original_size[1], original_size[0]),  # (H, W)
                mode='bilinear',
                align_corners=False
            )

            # Apply threshold for binary mask
            mask_binary = (mask_resized > threshold).float()

            # Convert mask to ComfyUI format [H, W]
            mask_np = mask_binary[0, 0].cpu().numpy()

            # Create output image
            if refine_foreground:
                # Apply mask to create RGBA with clean edges
                img_np = np.array(pil_image).astype(np.float32) / 255.0

                # Create RGBA image
                rgba_np = np.zeros((original_size[1], original_size[0], 4), dtype=np.float32)
                rgba_np[:, :, :3] = img_np
                rgba_np[:, :, 3] = mask_np

                # Convert to ComfyUI tensor [1, H, W, C]
                output_tensor = torch.from_numpy(rgba_np).unsqueeze(0)
            else:
                # Return original image
                output_tensor = img_tensor.unsqueeze(0)

            output_images.append(output_tensor)
            output_masks.append(torch.from_numpy(mask_np))

        # Stack batch
        batch_images = torch.cat(output_images, dim=0)
        batch_masks = torch.stack(output_masks, dim=0)

        logger.info(f"Background removal complete: {batch_size} images processed")

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

        return (batch_images, batch_masks)


NODE_CLASS_MAPPINGS = {
    "WaLaRemoveBackground": WaLaRemoveBackground,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaLaRemoveBackground": "WaLa Remove Background",
}
