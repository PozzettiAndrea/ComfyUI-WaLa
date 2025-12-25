"""
ComfyUI-WaLa PreStartup Script
- Copies example images from assets/ to ComfyUI input folder on startup
"""
import os
import shutil


def copy_example_assets():
    """Copy image files from assets/ directory to ComfyUI input directory."""
    try:
        import folder_paths

        input_folder = folder_paths.get_input_directory()
        custom_node_dir = os.path.dirname(os.path.abspath(__file__))

        assets_folder = os.path.join(custom_node_dir, "assets")
        if not os.path.exists(assets_folder):
            print(f"[ComfyUI-WaLa] Warning: assets folder not found at {assets_folder}")
            return

        # Only copy image files, skip hidden files and checkpoint folders
        image_extensions = {'.png', '.jpg', '.jpeg', '.webp'}
        copied_count = 0

        for file in os.listdir(assets_folder):
            # Skip hidden files and directories
            if file.startswith('.'):
                continue

            source_file = os.path.join(assets_folder, file)

            # Skip directories
            if os.path.isdir(source_file):
                continue

            # Check if it's an image file
            _, ext = os.path.splitext(file)
            if ext.lower() not in image_extensions:
                continue

            dest_file = os.path.join(input_folder, file)

            if not os.path.exists(dest_file):
                shutil.copy2(source_file, dest_file)
                copied_count += 1
                print(f"[ComfyUI-WaLa] Copied {file} to input/")

        if copied_count > 0:
            print(f"[ComfyUI-WaLa] [OK] Copied {copied_count} example image(s) to {input_folder}")
        else:
            print(f"[ComfyUI-WaLa] Example images already exist in {input_folder}")

    except Exception as e:
        print(f"[ComfyUI-WaLa] Error copying assets: {e}")


# Run on import
copy_example_assets()
