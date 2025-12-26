import logging
import urllib.error

import backoff
import torch
from mvdream_module import MVDreamModule
from latent_module import Trainer_Condition_Network
from huggingface_hub import hf_hub_download, logging as hf_logging
import os

# Enable huggingface_hub progress bars
hf_logging.set_verbosity_info()


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


def load_mvdream_model(
    pretrained_model_name_or_path,
    device=None,
    eval=True,
):

    if os.path.isfile(pretrained_model_name_or_path):
        checkpoint_path = pretrained_model_name_or_path
    else:
        # Check if model exists in ComfyUI models/wala folder
        models_dir = get_wala_models_dir()
        model_name = pretrained_model_name_or_path.split("/")[-1]
        local_model_dir = os.path.join(models_dir, model_name)
        checkpoint_path = os.path.join(local_model_dir, "checkpoint.ckpt")

        if os.path.exists(checkpoint_path):
            print(f"[ComfyUI-WaLa] Found local MVDream model: {local_model_dir}")
        else:
            # Download directly to local folder
            os.makedirs(local_model_dir, exist_ok=True)
            print(f"[ComfyUI-WaLa] Downloading MVDream {model_name} to {local_model_dir}...")

            hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="checkpoint.ckpt",
                local_dir=local_model_dir,
            )

            # Clean up .cache folder
            cache_dir = os.path.join(local_model_dir, ".cache")
            if os.path.exists(cache_dir):
                import shutil
                shutil.rmtree(cache_dir)

            print(f"[ComfyUI-WaLa] Download complete! Saved to {local_model_dir}")

    print(f"[ComfyUI-WaLa] Loading MVDream model...")
    model = MVDreamModule.load_from_checkpoint(checkpoint_path=checkpoint_path)

    if device is not None:
        model = model.to(device)
    if eval:
        model.eval()

    print(f"[ComfyUI-WaLa] MVDream model loaded successfully!")
    return model


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

def download_args_path(s3_path):
    from boto3.s3.transfer import TransferConfig
    import boto3
    import botocore
    import io
    import json

    max_concurrency = 10000
    multipart_size = 1024 * 1024 * 8
    boto3_config = botocore.config.Config(
        max_pool_connections=max_concurrency,
        s3={'max_queue_size': max_concurrency},
        connect_timeout=180,
        read_timeout=180,
        retries={'max_attempts': 10}
    )

    # Ensure the path starts with 's3://'
    if not s3_path.startswith("s3://"):
        raise ValueError("s3_path must start with 's3://'")

    # Remove 's3://' from the path and split by the first '/'
    s3_path_cleaned = s3_path[5:]
    bucket_name, file_path = s3_path_cleaned.split("/", 1)  # Split into bucket name and file path

    # Initialize boto3 S3 resource
    s3 = boto3.resource('s3', config=boto3_config, region_name='us-east-1')

    # Download the JSON file
    file = io.BytesIO()
    s3.Bucket(bucket_name).download_fileobj(file_path, file)
    
    # Move the file pointer to the beginning
    file.seek(0)
    
    # Load the JSON data
    data = json.load(file, object_hook=DotDict)
    
    return data

@backoff.on_exception(backoff.expo, (AttributeError, urllib.error.URLError), max_time=120)
def load_latent_model(
    json_path, 
    checkpoint_path,
    compile_model,
    device=None,
    eval=True,
):  
    args = download_args_path(json_path)
    #print(args)
    model = Trainer_Condition_Network.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location="cpu",
        args=args
    )

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