"""Inference nodes for WaLa 3D generation."""
import gc
import os
import tempfile
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from pytorch_lightning import seed_everything

import comfy.model_management as mm
import folder_paths

from .utils import logger, tensor_to_pil, pil_to_tensor, get_temp_output_dir


class WaLaSingleViewTo3D:
    """Generate 3D mesh from a single image using WaLa."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WALA_MODEL",),
                "image": ("IMAGE",),
            },
            "optional": {
                "scale": ("FLOAT", {"default": 1.8, "min": 0.5, "max": 5.0, "step": 0.1,
                                    "tooltip": "Classifier-free guidance scale. Higher = stronger adherence to input"}),
                "diffusion_rescale_timestep": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1,
                                                       "tooltip": "Diffusion rescale timestep for quality control"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
                "output_format": (["obj", "sdf"], {"default": "obj"}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh", "mesh_path")
    FUNCTION = "generate"
    CATEGORY = "WaLa"
    DESCRIPTION = """
Generate 3D mesh from a single image.

Parameters:
- model: WaLa model (use WaLa-SV-1B)
- image: Input image
- scale: Classifier-free guidance scale (default 1.8)
- diffusion_rescale_timestep: Quality control (default 5)
- seed: Random seed for reproducibility
- output_format: obj or sdf

Returns:
- mesh: Trimesh object for further processing
- mesh_path: Path to the generated mesh file
"""

    def generate(self, model, image, scale=1.8, diffusion_rescale_timestep=5, seed=42, output_format="obj"):
        import trimesh

        wala_model = model["model"]
        device = model["device"]

        # Set seed
        seed_everything(seed, workers=True)

        logger.info(f"Generating 3D from single view (scale={scale}, timestep={diffusion_rescale_timestep})")

        # Convert ComfyUI tensor to PIL
        pil_image = tensor_to_pil(image)

        # Save to temp file for WaLa
        temp_dir = get_temp_output_dir()
        temp_image_path = os.path.join(temp_dir, "input_image.png")
        pil_image.save(temp_image_path)

        # Prepare data
        from ..wala.dataset_utils import get_singleview_data, get_image_transform_latent_model

        image_transform = get_image_transform_latent_model()
        data = get_singleview_data(
            image_file=temp_image_path,
            image_transform=image_transform,
            device=device,
            image_over_white=False,
        )

        # Set inference params and run
        wala_model.set_inference_fusion_params(scale, diffusion_rescale_timestep)

        save_dir = Path(temp_dir) / "output"
        save_dir.mkdir(parents=True, exist_ok=True)

        output_path = wala_model.test_inference(
            data,
            data_idx=0,
            image_name="wala_output",
            save_dir=save_dir,
            output_format=output_format,
        )

        logger.info(f"3D mesh generated: {output_path}")

        # Load mesh for further processing
        if output_format == "obj":
            mesh = trimesh.load(output_path, process=False)
        else:
            mesh = None  # SDF format doesn't load to trimesh directly

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

        return (mesh, str(output_path))


class WaLaMultiViewTo3D:
    """Generate 3D mesh from 4 multi-view images using WaLa."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WALA_MODEL",),
                "image_0": ("IMAGE",),
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
            },
            "optional": {
                "view_0": ("INT", {"default": 0, "min": 0, "max": 100,
                                   "tooltip": "View index for first image (default 0)"}),
                "view_1": ("INT", {"default": 6, "min": 0, "max": 100,
                                   "tooltip": "View index for second image (default 6)"}),
                "view_2": ("INT", {"default": 10, "min": 0, "max": 100,
                                   "tooltip": "View index for third image (default 10)"}),
                "view_3": ("INT", {"default": 26, "min": 0, "max": 100,
                                   "tooltip": "View index for fourth image (default 26)"}),
                "scale": ("FLOAT", {"default": 1.3, "min": 0.5, "max": 5.0, "step": 0.1}),
                "diffusion_rescale_timestep": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
                "output_format": (["obj", "sdf"], {"default": "obj"}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh", "mesh_path")
    FUNCTION = "generate"
    CATEGORY = "WaLa"
    DESCRIPTION = """
Generate 3D mesh from 4 multi-view images.

Parameters:
- model: WaLa model (use WaLa-RGB4-1B)
- image_0-3: Four input images from different viewpoints
- view_0-3: View indices for each image (default: 0, 6, 10, 26)
- scale: Classifier-free guidance scale (default 1.3)
- diffusion_rescale_timestep: Quality control (default 5)

View index reference (elevation, azimuth):
- 0: (0, 0), 6: (0, 90), 10: (0, 150), 26: (30, 30)
- See WaLa documentation for full camera matrix
"""

    def generate(self, model, image_0, image_1, image_2, image_3,
                 view_0=0, view_1=6, view_2=10, view_3=26,
                 scale=1.3, diffusion_rescale_timestep=5, seed=42, output_format="obj"):
        import trimesh

        wala_model = model["model"]
        device = model["device"]

        seed_everything(seed, workers=True)

        logger.info(f"Generating 3D from multi-view (views={[view_0, view_1, view_2, view_3]}, scale={scale})")

        # Save images to temp files
        temp_dir = get_temp_output_dir()
        image_paths = []
        for i, img_tensor in enumerate([image_0, image_1, image_2, image_3]):
            pil_image = tensor_to_pil(img_tensor)
            path = os.path.join(temp_dir, f"mv_image_{i}.png")
            pil_image.save(path)
            image_paths.append(path)

        # Prepare data
        from ..wala.dataset_utils import get_multiview_data, get_image_transform_latent_model

        image_transform = get_image_transform_latent_model()
        views = [view_0, view_1, view_2, view_3]

        data = get_multiview_data(
            image_files=image_paths,
            views=views,
            image_transform=image_transform,
            device=device,
        )

        # Set inference params and run
        wala_model.set_inference_fusion_params(scale, diffusion_rescale_timestep)

        save_dir = Path(temp_dir) / "output"
        save_dir.mkdir(parents=True, exist_ok=True)

        output_path = wala_model.test_inference(
            data,
            data_idx=0,
            image_name="wala_mv_output",
            save_dir=save_dir,
            output_format=output_format,
        )

        logger.info(f"3D mesh generated: {output_path}")

        # Load mesh
        if output_format == "obj":
            mesh = trimesh.load(output_path, process=False)
        else:
            mesh = None

        gc.collect()
        torch.cuda.empty_cache()

        return (mesh, str(output_path))


class WaLaDepthMapTo3D:
    """Generate 3D mesh from depth maps using WaLa."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WALA_MODEL",),
                "depth_images": ("IMAGE",),
            },
            "optional": {
                "views": ("STRING", {"default": "3,6,10,26",
                                     "tooltip": "Comma-separated view indices. For 6 views use: 3,6,10,26,49,50"}),
                "scale": ("FLOAT", {"default": 1.3, "min": 0.5, "max": 5.0, "step": 0.1}),
                "diffusion_rescale_timestep": ("INT", {"default": 5, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
                "output_format": (["obj", "sdf"], {"default": "obj"}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh", "mesh_path")
    FUNCTION = "generate"
    CATEGORY = "WaLa"
    DESCRIPTION = """
Generate 3D mesh from depth map(s).

Parameters:
- model: WaLa model (use WaLa-DM1-1B, WaLa-DM4-1B, or WaLa-DM6-1B)
- depth_images: Batched depth map images (1, 4, or 6 images)
- views: Comma-separated view indices
  - For 4 views: "3,6,10,26"
  - For 6 views: "3,6,10,26,49,50"
  - For single view: "0"
- scale: Guidance scale (default 1.3, use 1.5 for 6 views)
- diffusion_rescale_timestep: Quality (default 5, use 10 for 6 views)
"""

    def generate(self, model, depth_images, views="3,6,10,26",
                 scale=1.3, diffusion_rescale_timestep=5, seed=42, output_format="obj"):
        import trimesh

        wala_model = model["model"]
        device = model["device"]

        seed_everything(seed, workers=True)

        # Parse view indices
        view_list = [int(v.strip()) for v in views.split(",")]
        num_views = depth_images.shape[0] if depth_images.dim() == 4 else 1

        logger.info(f"Generating 3D from {num_views} depth map(s) (views={view_list}, scale={scale})")

        temp_dir = get_temp_output_dir()

        # Save depth images
        image_paths = []
        if depth_images.dim() == 4:
            for i in range(depth_images.shape[0]):
                pil_image = tensor_to_pil(depth_images[i:i+1])
                path = os.path.join(temp_dir, f"depth_{i}.png")
                pil_image.save(path)
                image_paths.append(path)
        else:
            pil_image = tensor_to_pil(depth_images)
            path = os.path.join(temp_dir, "depth_0.png")
            pil_image.save(path)
            image_paths.append(path)

        from ..wala.dataset_utils import get_mv_dm_data, get_sv_dm_data, get_image_transform_latent_model

        image_transform = get_image_transform_latent_model()

        # Use appropriate data loader based on number of views
        if len(image_paths) == 1:
            data = get_sv_dm_data(
                image_file=image_paths[0],
                image_transform=image_transform,
                device=device,
                image_over_white=False,
            )
        else:
            data = get_mv_dm_data(
                image_files=image_paths,
                views=view_list[:len(image_paths)],
                image_transform=image_transform,
                device=device,
            )

        wala_model.set_inference_fusion_params(scale, diffusion_rescale_timestep)

        save_dir = Path(temp_dir) / "output"
        save_dir.mkdir(parents=True, exist_ok=True)

        output_path = wala_model.test_inference(
            data,
            data_idx=0,
            image_name="wala_dm_output",
            save_dir=save_dir,
            output_format=output_format,
        )

        logger.info(f"3D mesh generated: {output_path}")

        if output_format == "obj":
            mesh = trimesh.load(output_path, process=False)
        else:
            mesh = None

        gc.collect()
        torch.cuda.empty_cache()

        return (mesh, str(output_path))


class WaLaPointcloudTo3D:
    """Generate 3D mesh from pointcloud using WaLa."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WALA_MODEL",),
                "pointcloud_file": ("STRING", {"default": "",
                                               "tooltip": "Path to HDF5 (.h5df) pointcloud file with 'points' key"}),
            },
            "optional": {
                "scale": ("FLOAT", {"default": 1.3, "min": 0.5, "max": 5.0, "step": 0.1}),
                "diffusion_rescale_timestep": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
                "output_format": (["obj", "sdf"], {"default": "obj"}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh", "mesh_path")
    FUNCTION = "generate"
    CATEGORY = "WaLa"
    DESCRIPTION = """
Generate 3D mesh from a pointcloud.

Parameters:
- model: WaLa model (use WaLa-PC-1B)
- pointcloud_file: Path to HDF5 file containing pointcloud
  - Must have a 'points' key with shape (N, 3)
- scale: Guidance scale (default 1.3)
- diffusion_rescale_timestep: Quality control (default 8)

The HDF5 file should contain points in normalized coordinates.
"""

    def generate(self, model, pointcloud_file, scale=1.3, diffusion_rescale_timestep=8, seed=42, output_format="obj"):
        import trimesh

        if not pointcloud_file or not os.path.exists(pointcloud_file):
            raise ValueError(f"Pointcloud file not found: {pointcloud_file}")

        wala_model = model["model"]
        device = model["device"]

        seed_everything(seed, workers=True)

        logger.info(f"Generating 3D from pointcloud: {pointcloud_file}")

        from ..wala.dataset_utils import get_pointcloud_data

        data = get_pointcloud_data(
            pointcloud_file=pointcloud_file,
            device=device,
        )

        wala_model.set_inference_fusion_params(scale, diffusion_rescale_timestep)

        temp_dir = get_temp_output_dir()
        save_dir = Path(temp_dir) / "output"
        save_dir.mkdir(parents=True, exist_ok=True)

        output_path = wala_model.test_inference(
            data,
            data_idx=0,
            image_name="wala_pc_output",
            save_dir=save_dir,
            output_format=output_format,
        )

        logger.info(f"3D mesh generated: {output_path}")

        if output_format == "obj":
            mesh = trimesh.load(output_path, process=False)
        else:
            mesh = None

        gc.collect()
        torch.cuda.empty_cache()

        return (mesh, str(output_path))


class WaLaTextToDepthMaps:
    """Generate depth maps from text using WaLa MVDream."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mvdream_model": ("WALA_MVDREAM",),
                "prompt": ("STRING", {"default": "a ceramic cup", "multiline": True}),
            },
            "optional": {
                "num_frames": ("INT", {"default": 6, "min": 4, "max": 6, "step": 2,
                                       "tooltip": "Number of depth map views (4 or 6)"}),
                "image_size": ("INT", {"default": 256, "min": 128, "max": 512, "step": 64}),
                "step": ("INT", {"default": 50, "min": 10, "max": 100, "step": 5}),
                "guidance_scale": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 20.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("depth_maps", "view_indices")
    FUNCTION = "generate"
    CATEGORY = "WaLa"
    DESCRIPTION = """
Generate depth maps from text using MVDream.

Parameters:
- mvdream_model: WaLa MVDream model (use WaLa-MVDream-DM6)
- prompt: Text description of the object
- num_frames: Number of depth map views (4 or 6)
- image_size: Output image size
- step: Diffusion steps
- guidance_scale: Classifier-free guidance scale

The generated depth maps can be used with WaLa-DM4-1B or WaLa-DM6-1B
to create the final 3D mesh.
"""

    def generate(self, mvdream_model, prompt, num_frames=6, image_size=256, step=50, guidance_scale=10.0):
        model = mvdream_model["model"]

        logger.info(f"Generating {num_frames} depth maps from text: '{prompt}'")

        # View indices based on number of frames
        if num_frames == 6:
            testing_views = [3, 6, 10, 26, 49, 50]
        else:
            testing_views = [3, 6, 10, 26]

        images_np, image_views = model.inference_step(
            prompt=prompt,
            num_frames=num_frames,
            testing_views=testing_views,
            image_size=image_size,
            step=step,
            guidance_scale=guidance_scale,
        )

        logger.info(f"Generated {len(images_np)} depth maps")

        # Convert numpy images to ComfyUI tensor batch
        tensors = []
        for img_np in images_np:
            pil_img = Image.fromarray(img_np)
            tensor = pil_to_tensor(pil_img)
            tensors.append(tensor)

        # Stack into batch [B, H, W, C]
        batch_tensor = torch.cat(tensors, dim=0)

        # View indices as comma-separated string
        views_str = ",".join(map(str, image_views))

        gc.collect()
        torch.cuda.empty_cache()

        return (batch_tensor, views_str)


class WaLaTextToMultiView:
    """Generate multi-view RGB images from text using WaLa MVDream."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mvdream_model": ("WALA_MVDREAM",),
                "prompt": ("STRING", {"default": "a ceramic cup", "multiline": True}),
            },
            "optional": {
                "image_size": ("INT", {"default": 256, "min": 128, "max": 512, "step": 64}),
                "step": ("INT", {"default": 50, "min": 10, "max": 100, "step": 5}),
                "guidance_scale": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 20.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("multi_view_images", "view_indices")
    FUNCTION = "generate"
    CATEGORY = "WaLa"
    DESCRIPTION = """
Generate multi-view RGB images from text using MVDream.

Parameters:
- mvdream_model: WaLa MVDream model (use WaLa-MVDream-RGB4)
- prompt: Text description of the object
- image_size: Output image size
- step: Diffusion steps
- guidance_scale: Classifier-free guidance scale

The generated RGB images can be used with WaLa-RGB4-1B
to create the final 3D mesh.
"""

    def generate(self, mvdream_model, prompt, image_size=256, step=50, guidance_scale=10.0):
        model = mvdream_model["model"]

        logger.info(f"Generating 4 multi-view images from text: '{prompt}'")

        num_frames = 4
        testing_views = [0, 6, 10, 26]

        images_np, image_views = model.inference_step(
            prompt=prompt,
            num_frames=num_frames,
            testing_views=testing_views,
            image_size=image_size,
            step=step,
            guidance_scale=guidance_scale,
        )

        logger.info(f"Generated {len(images_np)} multi-view images")

        # Convert to ComfyUI tensor batch
        tensors = []
        for img_np in images_np:
            pil_img = Image.fromarray(img_np)
            tensor = pil_to_tensor(pil_img)
            tensors.append(tensor)

        batch_tensor = torch.cat(tensors, dim=0)
        views_str = ",".join(map(str, image_views))

        gc.collect()
        torch.cuda.empty_cache()

        return (batch_tensor, views_str)


NODE_CLASS_MAPPINGS = {
    "WaLaSingleViewTo3D": WaLaSingleViewTo3D,
    "WaLaMultiViewTo3D": WaLaMultiViewTo3D,
    "WaLaDepthMapTo3D": WaLaDepthMapTo3D,
    "WaLaPointcloudTo3D": WaLaPointcloudTo3D,
    "WaLaTextToDepthMaps": WaLaTextToDepthMaps,
    "WaLaTextToMultiView": WaLaTextToMultiView,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaLaSingleViewTo3D": "WaLa Single View to 3D",
    "WaLaMultiViewTo3D": "WaLa Multi-View to 3D",
    "WaLaDepthMapTo3D": "WaLa Depth Map to 3D",
    "WaLaPointcloudTo3D": "WaLa Pointcloud to 3D",
    "WaLaTextToDepthMaps": "WaLa Text to Depth Maps",
    "WaLaTextToMultiView": "WaLa Text to Multi-View",
}
