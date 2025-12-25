"""Utility functions for ComfyUI-WaLa."""
import logging
import torch
import numpy as np
from PIL import Image

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("[WaLa]")


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert ComfyUI image tensor to PIL Image.

    ComfyUI tensors are [B, H, W, C] with values in [0, 1].
    Takes the first image if batch size > 1.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image

    # Convert to numpy
    np_image = tensor.cpu().numpy()

    # Scale to 0-255
    np_image = (np_image * 255).astype(np.uint8)

    # Create PIL image
    if np_image.shape[-1] == 4:
        return Image.fromarray(np_image, mode='RGBA')
    elif np_image.shape[-1] == 3:
        return Image.fromarray(np_image, mode='RGB')
    else:
        return Image.fromarray(np_image.squeeze(), mode='L')


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert PIL Image to ComfyUI image tensor.

    Returns tensor of shape [1, H, W, C] with values in [0, 1].
    """
    # Convert to numpy
    np_image = np.array(image).astype(np.float32) / 255.0

    # Ensure 3 dimensions [H, W, C]
    if np_image.ndim == 2:
        np_image = np_image[:, :, np.newaxis]

    # Add batch dimension [1, H, W, C]
    tensor = torch.from_numpy(np_image).unsqueeze(0)

    return tensor


def get_wala_models_dir():
    """Get the directory for WaLa models."""
    import folder_paths
    import os

    models_dir = os.path.join(folder_paths.models_dir, "wala")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def get_temp_output_dir():
    """Get a temporary directory for WaLa outputs."""
    import folder_paths
    import os

    temp_dir = os.path.join(folder_paths.get_temp_directory(), "wala")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def process_image_with_mask(image_tensor: torch.Tensor, mask_tensor: torch.Tensor,
                            padding_percent: float = 0.1, target_size: int = 224,
                            background_color: str = "white") -> tuple:
    """
    Process image with mask: extract masked object, resize, center on canvas.

    Args:
        image_tensor: ComfyUI IMAGE tensor [B,H,W,C] with values in [0,1]
        mask_tensor: ComfyUI MASK tensor [H,W] or [B,H,W] with values in [0,1]
        padding_percent: Padding around object (0.1 = 10% on each side)
        target_size: Output size (224 for WaLa)
        background_color: "white" or "black" for canvas background

    Returns:
        tuple: (PIL Image ready for WaLa, ComfyUI tensor for preview)
    """
    bg_value = 1.0 if background_color == "white" else 0.0
    # Get first image if batched
    if image_tensor.dim() == 4:
        image_np = image_tensor[0].cpu().numpy()
    else:
        image_np = image_tensor.cpu().numpy()

    # Get mask as 2D array
    if mask_tensor.dim() == 3:
        mask_np = mask_tensor[0].cpu().numpy()
    else:
        mask_np = mask_tensor.cpu().numpy()

    h, w = mask_np.shape[:2]

    # Find bounding box of mask (where mask > 0.5)
    mask_binary = mask_np > 0.5
    rows = np.any(mask_binary, axis=1)
    cols = np.any(mask_binary, axis=0)

    if not np.any(rows) or not np.any(cols):
        # No mask content, use full image
        y_min, y_max = 0, h
        x_min, x_max = 0, w
    else:
        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]
        y_min, y_max = y_indices[0], y_indices[-1] + 1
        x_min, x_max = x_indices[0], x_indices[-1] + 1

    # Crop to bbox
    bbox_h = y_max - y_min
    bbox_w = x_max - x_min
    cropped_image = image_np[y_min:y_max, x_min:x_max]
    cropped_mask = mask_np[y_min:y_max, x_min:x_max]

    # Expand mask to 3 channels
    if cropped_mask.ndim == 2:
        cropped_mask = cropped_mask[:, :, np.newaxis]

    # Calculate object size (fits in center with padding on all sides)
    # padding_percent is per side, so object takes (1 - 2*padding) of the canvas
    object_size = int(target_size * (1 - padding_percent * 2))

    logger.info(f"Mask bbox: {bbox_w}x{bbox_h}, padding: {padding_percent*100:.0f}%, object_size: {object_size}px in {target_size}px canvas")

    # Make cropped region square by padding with background color
    max_dim = max(bbox_h, bbox_w)
    square_image = np.full((max_dim, max_dim, 3), bg_value, dtype=np.float32)
    square_mask = np.zeros((max_dim, max_dim, 1), dtype=np.float32)  # No mask

    # Center the crop in the square
    y_offset = (max_dim - bbox_h) // 2
    x_offset = (max_dim - bbox_w) // 2
    square_image[y_offset:y_offset+bbox_h, x_offset:x_offset+bbox_w] = cropped_image
    square_mask[y_offset:y_offset+bbox_h, x_offset:x_offset+bbox_w] = cropped_mask

    # Composite on background: bg where mask=0, image where mask=1
    composited = np.full_like(square_image, bg_value)
    bg_fill = np.full_like(cropped_image, bg_value)
    composited[y_offset:y_offset+bbox_h, x_offset:x_offset+bbox_w] = (
        bg_fill * (1 - cropped_mask) + cropped_image * cropped_mask
    )

    # Convert to PIL, resize to object_size
    composited_uint8 = (composited * 255).astype(np.uint8)
    object_pil = Image.fromarray(composited_uint8, mode='RGB')
    object_pil = object_pil.resize((object_size, object_size), Image.Resampling.BILINEAR)

    # Create canvas with background color and paste object in center
    bg_color = (255, 255, 255) if background_color == "white" else (0, 0, 0)
    canvas = Image.new('RGB', (target_size, target_size), bg_color)
    paste_offset = (target_size - object_size) // 2
    canvas.paste(object_pil, (paste_offset, paste_offset))

    # Convert back to tensor for preview output
    preview_np = np.array(canvas).astype(np.float32) / 255.0
    preview_tensor = torch.from_numpy(preview_np).unsqueeze(0)  # [1, H, W, C]

    return canvas, preview_tensor
