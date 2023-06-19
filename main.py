import os, sys

sys.path.append(os.path.join(os.getcwd(), "Grounded_Segment_Anything"))
sys.path.append(os.path.join(os.getcwd(), "Grounded_Segment_Anything/GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "Grounded_Segment_Anything/segment_anything"))

import argparse
import copy

from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

import supervision as sv


import cv2
import numpy as np
import matplotlib.pyplot as plt


# diffusers
import PIL
import torch
from pathlib import Path

from code.builders import load_grounded_dino_model_hf, load_segment_anything_model, load_stable_diffusion_inpaint_pipeline
from code.data import download_image
from code.tools import detect_bounding_box


def main():

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device1 = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    device1 = torch.device('cpu')

    # Get the Grounded DINO trained vit
    groundingdino_model = load_grounded_dino_model_hf(device=device1)

    # Get the segment anything model
    sam = load_segment_anything_model(device=device1)

    # Get the stable diffusion inpaint pipeline
    stable_diffusion_inpaint_pipeline = load_stable_diffusion_inpaint_pipeline(device=device1)

    # Get the image
    image_file_path = Path('assets/groundingdino_example.jpg')
    if not image_file_path.exists():
        download_image(save_path = image_file_path)
    image_source, image = load_image(image_file_path)
    

    # Use the Grounded DINO model to predict the bounding box
    annotated_frame, detected_boxes = detect_bounding_box(image, image_source, text_prompt="bench", model=groundingdino_model, device=device1)
    annotated_frame = Image.fromarray(annotated_frame)
    annotated_frame.save("assets/groundingdino_example_annotated.jpg")
    print("Detected boxes: ", detected_boxes)
    print(type(annotated_frame))




if __name__ == '__main__':
    main()