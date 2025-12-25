import logging
import torch
from latent_module import (
    Trainer_Condition_Network,
)
from networks.callbacks import EMACallback
import os
import json
import backoff
import urllib.error
from huggingface_hub import hf_hub_download, logging as hf_logging
from huggingface_hub.utils import enable_progress_bars, are_progress_bars_disabled
import re

# Force enable huggingface_hub progress bars
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # Disable hf_transfer to see progress
hf_logging.set_verbosity_info()
enable_progress_bars()


class DotDict(dict):
    def __getattr__(self, attr_):
        try:
            return self[attr_]
        except KeyError:
            print(f"'DotDict' object has no attribute '{attr_}'")

    def __setattr__(self, attr_, value):
        self[attr_] = value

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __deepcopy__(self, memo):
        return DotDict(copy.deepcopy(dict(self), memo))


@backoff.on_exception(
    backoff.expo, (AttributeError, urllib.error.URLError), max_time=120
)
def load_latent_model(
    json_path,
    checkpoint_path,
    compile_model,
    device=None,
    eval=True,
):
    with open(json_path, "r") as file:
        args = json.load(file, object_hook=DotDict)

    # Load checkpoint manually to patch state_dict if needed
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    # Patch: Remove _orig_mod. prefix if present (from torch.compile)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("autoencoder._orig_mod."):
            new_state_dict[k.replace("autoencoder._orig_mod.", "autoencoder.")] = v
        else:
            new_state_dict[k] = v
    checkpoint["state_dict"] = new_state_dict

    # Now load the model using the patched checkpoint
    model = Trainer_Condition_Network.load_from_checkpoint(
        checkpoint_path=checkpoint_path, map_location="cpu", args=args, strict=False
    )
    # Overwrite model state_dict with patched one (to ensure correct keys)
    model.load_state_dict(new_state_dict, strict=False)

    if hasattr(model, "ema_state_dict") and model.ema_state_dict is not None:
        # load EMA weights
        ema = EMACallback(decay=0.9999)
        ema.reload_weight = model.ema_state_dict
        ema.reload_weight_for_pl_module(model)
        ema.copy_to_pl_module(model)

    if compile_model:
        logging.info("Compiling models...")
        model.network.training_losses = torch.compile(model.network.training_losses)
        model.network.inference = torch.compile(model.network.inference)
        if hasattr(model, "clip_model"):
            model.clip_model.forward = torch.compile(model.clip_model.forward)
        logging.info("Done Compiling models...")

    if device is not None:
        model = model.to(device)
    if eval:
        model.eval()

    return model

def get_wala_models_dir():
    """Get the WaLa models directory in ComfyUI's models folder."""
    try:
        import folder_paths
        models_dir = os.path.join(folder_paths.models_dir, "wala")
    except ImportError:
        # Fallback if folder_paths not available
        models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "wala")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


class Model:

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        if os.path.isfile(pretrained_model_name_or_path):
            checkpoint_path = pretrained_model_name_or_path
            json_path = os.path.dirname(checkpoint_path) + "/args.json"
        else:
            # Check if model exists in ComfyUI models/wala folder
            models_dir = get_wala_models_dir()
            # Convert repo_id like "ADSKAILab/WaLa-SV-1B" to folder name "WaLa-SV-1B"
            model_name = pretrained_model_name_or_path.split("/")[-1]
            local_model_dir = os.path.join(models_dir, model_name)

            json_path = os.path.join(local_model_dir, "args.json")
            checkpoint_path = os.path.join(local_model_dir, "checkpoint.ckpt")

            # Check if already downloaded locally
            if os.path.exists(json_path) and os.path.exists(checkpoint_path):
                print(f"[ComfyUI-WaLa] Found local model: {local_model_dir}")
            else:
                # Download directly to local folder
                os.makedirs(local_model_dir, exist_ok=True)
                print(f"[ComfyUI-WaLa] Downloading {model_name} to {local_model_dir}...")

                print(f"[ComfyUI-WaLa] Downloading args.json...")
                hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename="args.json",
                    local_dir=local_model_dir,
                    local_dir_use_symlinks=False,
                )

                print(f"[ComfyUI-WaLa] Downloading checkpoint.ckpt (~2-4GB)...")
                hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename="checkpoint.ckpt",
                    local_dir=local_model_dir,
                    local_dir_use_symlinks=False,
                )

                print(f"[ComfyUI-WaLa] Download complete! Saved to {local_model_dir}")

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        print(f"[ComfyUI-WaLa] Loading model from {os.path.dirname(checkpoint_path)}...")
        model = load_latent_model(
            json_path,
            checkpoint_path,
            compile_model=False,
            device=device,
        )
        print(f"[ComfyUI-WaLa] Model loaded successfully!")
        return model

########################################################


import boto3
from urllib.parse import urlparse

class Model_internal:
    @classmethod
    def from_pretrained(
        cls,
        model_config_uri: str,
        model_checkpoint_uri: str,
        local_dir: str = "/tmp/model_ckpt"
    ):
        def is_s3_uri(uri):
            return uri.startswith("s3://")

        def download_from_s3(s3_uri, local_path):
            parsed = urlparse(s3_uri)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            s3 = boto3.client('s3')
            s3.download_file(bucket, key, local_path)

        os.makedirs(local_dir, exist_ok=True)

        # Handle config
        if is_s3_uri(model_config_uri):
            local_config_path = os.path.join(local_dir, "args.json")
            download_from_s3(model_config_uri, local_config_path)
        else:
            local_config_path = model_config_uri

        # Handle checkpoint
        if is_s3_uri(model_checkpoint_uri):
            local_ckpt_path = os.path.join(local_dir, "checkpoint.ckpt")
            download_from_s3(model_checkpoint_uri, local_ckpt_path)
        else:
            local_ckpt_path = model_checkpoint_uri

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        model = load_latent_model(
            local_config_path,
            local_ckpt_path,
            compile_model=False,
            device=device,
        )
        return model