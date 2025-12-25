#!/usr/bin/env python3
"""
ComfyUI-WaLa Installer

Automatically installs spconv and other dependencies based on detected
PyTorch, CUDA, and Python versions.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def get_python_version():
    """Get Python version as string like '310' for 3.10."""
    return f"{sys.version_info.major}{sys.version_info.minor}"


def get_torch_info():
    """Get PyTorch and CUDA version info."""
    try:
        import torch
        torch_version = torch.__version__.split('+')[0]  # e.g., "2.8.0"
        cuda_version = torch.version.cuda  # e.g., "12.8" or None
        return torch_version, cuda_version
    except ImportError:
        return None, None


def parse_cuda_version(cuda_version):
    """Parse CUDA version string to tuple (major, minor)."""
    if not cuda_version:
        return None
    parts = cuda_version.split('.')
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return (major, minor)
    except (ValueError, IndexError):
        return None


def get_spconv_package(cuda_version):
    """
    Determine the appropriate spconv package for the CUDA version.

    Available packages:
    - spconv-cu120 (CUDA 12.0+)
    - spconv-cu117 (CUDA 11.7-11.8)
    - spconv-cu114 (CUDA 11.4-11.6)
    - spconv-cu113 (CUDA 11.3)
    - spconv-cu102 (CUDA 10.2)

    CUDA minor version compatibility allows using slightly older spconv builds.
    """
    cuda_tuple = parse_cuda_version(cuda_version)
    if not cuda_tuple:
        return None

    major, minor = cuda_tuple

    # CUDA 12.x -> use spconv-cu120
    if major >= 12:
        return "spconv-cu120"
    # CUDA 11.7+ -> use spconv-cu117
    elif major == 11 and minor >= 7:
        return "spconv-cu117"
    # CUDA 11.4-11.6 -> use spconv-cu114
    elif major == 11 and minor >= 4:
        return "spconv-cu114"
    # CUDA 11.3 -> use spconv-cu113
    elif major == 11 and minor == 3:
        return "spconv-cu113"
    # CUDA 10.2 -> use spconv-cu102
    elif major == 10 and minor >= 2:
        return "spconv-cu102"
    else:
        return None


def install_spconv():
    """Install spconv with automatic CUDA version detection."""
    print("\n" + "="*60)
    print("ComfyUI-WaLa: Installing spconv")
    print("="*60 + "\n")

    # Check if already installed
    try:
        import spconv
        print(f"[ComfyUI-WaLa] spconv already installed")
        return True
    except ImportError:
        pass

    # Get environment info
    torch_version, cuda_version = get_torch_info()
    python_version = get_python_version()

    print(f"[ComfyUI-WaLa] Detected environment:")
    print(f"  Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"  PyTorch: {torch_version or 'not installed'}")
    print(f"  CUDA: {cuda_version or 'not available'}")
    print(f"  Platform: {platform.system()} {platform.machine()}")

    if not torch_version:
        print("[ComfyUI-WaLa] ERROR: PyTorch not installed. Please install PyTorch first.")
        return False

    if not cuda_version:
        print("[ComfyUI-WaLa] ERROR: CUDA not available. spconv requires CUDA.")
        return False

    # Get appropriate spconv package
    spconv_package = get_spconv_package(cuda_version)

    if not spconv_package:
        print(f"[ComfyUI-WaLa] ERROR: No spconv package available for CUDA {cuda_version}")
        print("[ComfyUI-WaLa] Supported CUDA versions: 10.2, 11.3, 11.4+, 11.7+, 12.0+")
        return False

    print(f"[ComfyUI-WaLa] Installing {spconv_package} for CUDA {cuda_version}...")

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            spconv_package
        ])
        print(f"[ComfyUI-WaLa] {spconv_package} installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ComfyUI-WaLa] Failed to install {spconv_package}: {e}")

        # Try fallback to older version if 12.x failed
        cuda_tuple = parse_cuda_version(cuda_version)
        if cuda_tuple and cuda_tuple[0] >= 12:
            print("[ComfyUI-WaLa] Trying spconv-cu117 as fallback...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "spconv-cu117"
                ])
                print("[ComfyUI-WaLa] spconv-cu117 installed successfully!")
                return True
            except subprocess.CalledProcessError:
                pass

        return False


def install_requirements():
    """Install requirements from requirements.txt."""
    print("\n" + "="*60)
    print("ComfyUI-WaLa: Installing Requirements")
    print("="*60 + "\n")

    requirements_path = Path(__file__).parent / "requirements.txt"

    if requirements_path.exists():
        print(f"[ComfyUI-WaLa] Installing from {requirements_path}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "-r", str(requirements_path)
            ])
            print("[ComfyUI-WaLa] Requirements installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ComfyUI-WaLa] Failed to install requirements: {e}")
            return False
    else:
        print("[ComfyUI-WaLa] No requirements.txt found")
        return True


def main():
    """Main installation routine."""
    print("\n" + "="*60)
    print("ComfyUI-WaLa Installer")
    print("="*60 + "\n")

    success = True

    # Install basic requirements first
    if not install_requirements():
        success = False

    # Install spconv
    if not install_spconv():
        print("\n[ComfyUI-WaLa] WARNING: spconv installation failed!")
        print("[ComfyUI-WaLa] WaLa requires spconv for sparse convolutions.")
        print("[ComfyUI-WaLa] You can try installing it manually:")
        print("  pip install spconv-cu120  # For CUDA 12.x")
        print("  pip install spconv-cu117  # For CUDA 11.7+")
        print("  pip install spconv-cu114  # For CUDA 11.4+")
        success = False

    if success:
        print("\n" + "="*60)
        print("[ComfyUI-WaLa] Installation completed successfully!")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("[ComfyUI-WaLa] Installation completed with errors.")
        print("[ComfyUI-WaLa] Please check the messages above and install missing packages manually.")
        print("="*60 + "\n")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
