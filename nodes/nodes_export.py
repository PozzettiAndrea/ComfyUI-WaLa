"""Export and render nodes for WaLa 3D meshes."""
import os
import torch
import numpy as np
from datetime import datetime

import folder_paths

from .utils import logger


class WaLaExportOBJ:
    """Export mesh to OBJ file."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
            "optional": {
                "filename_prefix": ("STRING", {"default": "wala"}),
                "target_num_faces": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1000,
                                             "tooltip": "Target face count for simplification. 0 = no simplification"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("obj_path",)
    FUNCTION = "export"
    CATEGORY = "WaLa"
    OUTPUT_NODE = True
    DESCRIPTION = """
Export mesh to OBJ file.

Parameters:
- trimesh: The 3D mesh to export
- filename_prefix: Prefix for output filename
- target_num_faces: Target face count for mesh simplification (0 = no simplification)

Output OBJ is saved to ComfyUI output folder.
"""

    def export(self, trimesh, filename_prefix="wala", target_num_faces=0):
        if trimesh is None:
            raise ValueError("No mesh to export. Make sure you're using 'obj' output format.")

        logger.info(f"Exporting mesh to OBJ")

        # Simplify if requested
        if target_num_faces > 0 and len(trimesh.faces) > target_num_faces:
            try:
                import open3d as o3d

                # Convert to open3d mesh
                o3d_mesh = o3d.geometry.TriangleMesh()
                o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh.vertices)
                o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh.faces)

                # Simplify
                simplified = o3d_mesh.simplify_quadric_decimation(target_num_faces)

                # Convert back to trimesh
                import trimesh as Trimesh
                trimesh = Trimesh.Trimesh(
                    vertices=np.asarray(simplified.vertices),
                    faces=np.asarray(simplified.triangles),
                    process=False
                )
                logger.info(f"Simplified mesh to {len(trimesh.faces)} faces")
            except ImportError:
                logger.warning("open3d not installed, skipping simplification")

        # Generate filename with timestamp
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.obj"

        # Save to output folder
        output_dir = folder_paths.get_output_directory()
        output_path = os.path.join(output_dir, filename)

        trimesh.export(output_path)

        logger.info(f"OBJ exported to: {output_path}")

        return (output_path,)


class WaLaRenderPreview:
    """Render preview images of a mesh."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
            "optional": {
                "num_views": ("INT", {"default": 8, "min": 1, "max": 36, "step": 1}),
                "resolution": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 128}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview_images",)
    FUNCTION = "render"
    CATEGORY = "WaLa"
    DESCRIPTION = """
Render preview images of the 3D mesh.

Parameters:
- trimesh: The 3D mesh geometry
- num_views: Number of views to render (rotating around object)
- resolution: Render resolution
"""

    def render(self, trimesh, num_views=8, resolution=512):
        if trimesh is None:
            raise ValueError("No mesh to render. Make sure you're using 'obj' output format.")

        import pyrender
        import math

        logger.info(f"Rendering {num_views} preview images at {resolution}px")

        # Create pyrender scene
        scene = pyrender.Scene(bg_color=[0.1, 0.1, 0.1, 1.0])

        # Create mesh for pyrender
        mesh = pyrender.Mesh.from_trimesh(trimesh)
        scene.add(mesh)

        # Setup camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 5.0)

        # Calculate camera distance based on mesh bounds
        bounds = trimesh.bounds
        center = (bounds[0] + bounds[1]) / 2
        extent = np.linalg.norm(bounds[1] - bounds[0])
        distance = extent * 2.0

        # Render from multiple views
        frames = []
        renderer = pyrender.OffscreenRenderer(resolution, resolution)

        for i in range(num_views):
            angle = 2 * math.pi * i / num_views

            # Camera position
            cam_pos = np.array([
                center[0] + distance * math.sin(angle),
                center[1] + 0.3 * distance,
                center[2] + distance * math.cos(angle)
            ])

            # Look at center
            forward = center - cam_pos
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, np.array([0, 1, 0]))
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)

            camera_pose = np.eye(4)
            camera_pose[:3, 0] = right
            camera_pose[:3, 1] = up
            camera_pose[:3, 2] = -forward
            camera_pose[:3, 3] = cam_pos

            # Add camera to scene
            cam_node = scene.add(camera, pose=camera_pose)

            # Add light
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
            light_node = scene.add(light, pose=camera_pose)

            # Render
            color, _ = renderer.render(scene)
            frames.append(color)

            # Remove camera and light for next view
            scene.remove_node(cam_node)
            scene.remove_node(light_node)

        renderer.delete()

        # Convert to tensor batch [N, H, W, C]
        frames_np = np.stack(frames, axis=0)
        frames_tensor = torch.from_numpy(frames_np).float() / 255.0

        return (frames_tensor,)


class WaLaRenderVideo:
    """Render a rotating video of the mesh."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
            "optional": {
                "num_frames": ("INT", {"default": 60, "min": 10, "max": 360, "step": 10}),
                "fps": ("INT", {"default": 15, "min": 1, "max": 60, "step": 1}),
                "resolution": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 128}),
                "filename_prefix": ("STRING", {"default": "wala_video"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "render_video"
    CATEGORY = "WaLa"
    OUTPUT_NODE = True
    DESCRIPTION = """
Render a rotating video of the 3D mesh.

Parameters:
- trimesh: The 3D mesh geometry
- num_frames: Number of frames in the video
- fps: Frames per second
- resolution: Render resolution
- filename_prefix: Prefix for output filename
"""

    def render_video(self, trimesh, num_frames=60, fps=15, resolution=512, filename_prefix="wala_video"):
        if trimesh is None:
            raise ValueError("No mesh to render. Make sure you're using 'obj' output format.")

        import pyrender
        import imageio
        import math

        logger.info(f"Rendering video ({num_frames} frames at {fps}fps)...")

        # Create pyrender scene
        scene = pyrender.Scene(bg_color=[0.1, 0.1, 0.1, 1.0])

        # Create mesh for pyrender
        pyrender_mesh = pyrender.Mesh.from_trimesh(trimesh)
        scene.add(pyrender_mesh)

        # Setup camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 5.0)

        # Calculate camera distance
        bounds = trimesh.bounds
        center = (bounds[0] + bounds[1]) / 2
        extent = np.linalg.norm(bounds[1] - bounds[0])
        distance = extent * 2.0

        # Render frames
        frames = []
        renderer = pyrender.OffscreenRenderer(resolution, resolution)

        for i in range(num_frames):
            angle = 2 * math.pi * i / num_frames

            cam_pos = np.array([
                center[0] + distance * math.sin(angle),
                center[1] + 0.3 * distance,
                center[2] + distance * math.cos(angle)
            ])

            forward = center - cam_pos
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, np.array([0, 1, 0]))
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)

            camera_pose = np.eye(4)
            camera_pose[:3, 0] = right
            camera_pose[:3, 1] = up
            camera_pose[:3, 2] = -forward
            camera_pose[:3, 3] = cam_pos

            cam_node = scene.add(camera, pose=camera_pose)
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
            light_node = scene.add(light, pose=camera_pose)

            color, _ = renderer.render(scene)
            frames.append(color)

            scene.remove_node(cam_node)
            scene.remove_node(light_node)

        renderer.delete()

        # Generate filename
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.mp4"

        # Save to output folder
        output_dir = folder_paths.get_output_directory()
        output_path = os.path.join(output_dir, filename)

        imageio.mimsave(output_path, frames, fps=fps)

        logger.info(f"Video saved to: {output_path}")

        torch.cuda.empty_cache()

        return (output_path,)


NODE_CLASS_MAPPINGS = {
    "WaLaExportOBJ": WaLaExportOBJ,
    "WaLaRenderPreview": WaLaRenderPreview,
    "WaLaRenderVideo": WaLaRenderVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WaLaExportOBJ": "WaLa Export OBJ",
    "WaLaRenderPreview": "WaLa Render Preview",
    "WaLaRenderVideo": "WaLa Render Video",
}
