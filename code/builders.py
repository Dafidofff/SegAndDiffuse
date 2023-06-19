import torch

# Grounding DINO
from huggingface_hub import hf_hub_download
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.slconfig import SLConfig

# segment anything
from segment_anything import build_sam, SamPredictor 

# stable diffusion
from diffusers import StableDiffusionInpaintPipeline


def load_grounded_dino_model_hf(
        repo_id: str = "ShilongLiu/GroundingDINO", 
        filename: str = "groundingdino_swinb_cogcoor.pth", 
        ckpt_config_filename: str = "GroundingDINO_SwinB.cfg.py", 
        device='cpu'
    ):
    """
    Loads the Grounded DINO Vit model from HuggingFace Hub

    When creating this repo the following are the default values:
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

    Args:   
        repo_id (str): The HuggingFace Hub repo id
        filename (str): The filename of the checkpoint
        ckpt_config_filename (str): The filename of the config file
        device (str): The device to load the model on

    Returns:
        model (nn.Module): The loaded model

    """

    # Load the configs from HuggingFace Hub 
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    # Load the config and build model
    args = SLConfig.fromfile(cache_config_file) 
    args.device = device
    model = build_model(args)
    
    # Load the checkpoint from HuggingFace Hub
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)

    # Freeze model
    _ = model.eval()
    return model  


def load_segment_anything_model(ckpt_path = 'ckpts/sam_vit_h_4b8939.pth', device: str = 'cpu'):
    """
    Loads the Segment Anything Model.

    Args:
        ckpt_name (str): The name of the checkpoint to load.
        device (str): The device to load the model on. Default is 'cpu'.

    Returns:
        sam_predictor (SamPredictor): The loaded Segment Anything Model.
    """
    return SamPredictor(build_sam(checkpoint=ckpt_path).to(device))


def load_stable_diffusion_inpaint_pipeline(device: str = 'cpu'):
    sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    ).to(device)