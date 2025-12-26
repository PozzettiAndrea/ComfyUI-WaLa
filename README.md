# ComfyUI-WaLa

ComfyUI custom nodes for [WaLa](https://github.com/AutodeskAILab/WaLa) - a billion-parameter 3D generative model.

> **Note:** Only **single-view image** and **depth map** inputs are tested. Multi-view, pointcloud, and text-to-3D modes are included but not actively maintained. For those use cases, consider [TRELLIS2](https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2) which produces significantly better results.

## Workflows

Two workflows are implemented and tested:

### Single Image to 3D
![Single Image Workflow](docs/single_image.png)

### Depth Map to 3D
![Depth Map Workflow](docs/single_depth.png)

## Supported Models

### WaLa Models
- `ADSKAILab/WaLa-SV-1B` - Single-view
- `ADSKAILab/WaLa-DM1-1B` - Single depth map

## Optimal Parameters

| Model | Scale | Timestep |
|-------|-------|----------|
| Single-View RGB | 1.8 | 5 |
| Multi-View RGB | 1.3 | 5 |
| Multi-View Depth (4) | 1.3 | 5 |
| Multi-View Depth (6) | 1.5 | 10 |
| Text to 3D | 1.5 | 10 |
| Pointcloud | 1.3 | 8 |

## Installation

Use ComfyUI Manager, or clone this repo then run "pip install -r requirements.txt" and "python install.py".

## Credits

- [WaLa](https://github.com/AutodeskAILab/WaLa) by Autodesk AI Lab

## License

This project wraps the WaLa model. Please refer to the [WaLa repository](https://github.com/AutodeskAILab/WaLa) for licensing information.
