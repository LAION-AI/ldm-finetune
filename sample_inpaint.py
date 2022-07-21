from pathlib import Path
import typing

from PIL import Image 

import os

from guided_diffusion.inpaint_util import sample_inpaint, prepare_inpaint_models

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF

os.environ[
    "TOKENIZERS_PARALLELISM"
] = "false"  # required to avoid errors with transformers lib

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=str, default="")
    parser.add_argument("--negative", type=str, default="")
    parser.add_argument("--init_image", type=str, default=None)
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--init_skip_fraction", type=float, default=0.0)
    parser.add_argument("--aesthetic_rating", type=int, default=9)
    parser.add_argument("--aesthetic_weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--intermediate_outputs", type=bool, default=False)
    parser.add_argument("--model_path", type=str, default="simulacra_540K.pt")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    prompts = args.prompts
    negative = args.negative
    init_image = args.init_image
    mask = args.mask
    guidance_scale = args.guidance_scale
    steps = args.steps
    batch_size = args.batch_size
    width = args.width
    height = args.height
    init_skip_fraction = args.init_skip_fraction
    aesthetic_rating = args.aesthetic_rating
    aesthetic_weight = args.aesthetic_weight
    seed = args.seed
    intermediate_outputs = args.intermediate_outputs
    model_path = args.model_path

    inpaint_models = prepare_inpaint_models(inpaint_model_path=model_path, device="cuda", use_fp16=False)

    if ".txt" in prompts and Path(prompts).exists():
        with open(prompts, "r") as f:
            prompts = f.readlines()
        print(f"Read {len(prompts)} prompts from {prompts}")
    else:
        prompts = [prompts]
    
    for prompt in prompts:
        print(f"Generating prompt: {prompt}")
        generations = list(
            sample_inpaint(
                prompt=prompt,
                negative=negative,
                init_image=init_image,
                mask=mask,
                guidance_scale=guidance_scale,
                steps=steps,
                batch_size=batch_size,
                width=width,
                height=height,
                init_skip_fraction=init_skip_fraction,
                aesthetic_rating=aesthetic_rating,
                aesthetic_weight=aesthetic_weight,
                seed=seed,
                intermediate_outputs=intermediate_outputs,
                loaded_models=inpaint_models,
            )
        )