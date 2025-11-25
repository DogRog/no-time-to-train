import copy
import os
import torch
from transformers import AutoModel
from huggingface_hub import snapshot_download

def _resolve_encoder_checkpoint(hf_model_id: str, encoder_defaults: dict) -> str:
    """Ensure the encoder checkpoint is a valid Hugging Face repo id or local directory.

    If a legacy *.pth path is provided, download the corresponding repo and return the local directory.
    """

    # Accept already-downloaded directories or valid repo ids directly.
    if os.path.isdir(hf_model_id):
        return hf_model_id

    # If a legacy checkpoint file path is supplied, download the HF model next to it.
    if os.path.isfile(hf_model_id):
        repo_id = encoder_defaults.get("hf_model_name")
        if repo_id is None:
            raise ValueError(
                f"Cannot resolve Hugging Face model for checkpoint '{hf_model_id}' "
                "because the preset does not specify a 'hf_model_name'."
            )

        target_dir = os.path.splitext(hf_model_id)[0] + "_hf"
        if not os.path.isdir(target_dir) or not os.listdir(target_dir):
            print(f"Downloading {repo_id} to {target_dir}...")
            snapshot_download(repo_id=repo_id, local_dir=target_dir)
        return target_dir

    # Otherwise assume it's a repo id string understood by Transformers.
    return hf_model_id

def build_encoder(encoder_cfg, encoder_ckpt_path, encoder_predefined_cfgs, device):
    encoder_cfg = copy.deepcopy(encoder_cfg)
    encoder_name = encoder_cfg.pop("name")
    encoder_defaults = copy.deepcopy(encoder_predefined_cfgs.get(encoder_name, {}))

    if not encoder_defaults:
        raise KeyError(f"Unsupported encoder preset '{encoder_name}'.")

    encoder_img_size = encoder_cfg.get("img_size", encoder_defaults.get("img_size"))
    encoder_patch_size = encoder_cfg.get("patch_size", encoder_defaults.get("patch_size"))
    encoder_hw = encoder_img_size // encoder_patch_size

    hf_model_id = encoder_cfg.pop("hf_model_name", None)
    if hf_model_id is None:
        if encoder_ckpt_path is not None:
            hf_model_id = encoder_ckpt_path
        else:
            hf_model_id = encoder_defaults.get("hf_model_name")

    if hf_model_id is None:
        raise ValueError(
            "A Hugging Face Dinov2 repository ID or local directory (hf_model_name/encoder_ckpt_path) must be provided."
        )

    hf_model_id = _resolve_encoder_checkpoint(hf_model_id, encoder_defaults)
    encoder = AutoModel.from_pretrained(hf_model_id)

    encoder.to(device)
    encoder.eval()
    
    return encoder, encoder_img_size, encoder_patch_size, encoder_hw

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Reference: MoCo v2
    Gathers tensors from all GPUs and concatenates them.
    Works with single GPU or CPU training as well.
    """
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor
        
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output