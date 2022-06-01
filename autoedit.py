import argparse
import io
import json
import os
from typing import Tuple

import numpy as np
import requests
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm

import wandb
from clip_custom import clip
from encoders.modules import BERTEmbedder
from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


BASE_DIR = os.environ["BASE_DIR"]
if BASE_DIR == "":
    BASE_DIR = "output"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(BASE_DIR + "_npy", exist_ok=True)
print(f"Using base directory {BASE_DIR}")


def load_aesthetic_vit_l_14_embed(
    rating: int = 9, embed_dir: str = "aesthetic-predictor/vit_l_14_embeddings"
) -> torch.Tensor:
    assert rating in [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ], "rating must be in [1, 2, 3, 4, 5, 6, 7, 8, 9]"
    embed_path = os.path.join(embed_dir, f"rating{rating}.npy")
    text_emb_clip_aesthetic = np.load(embed_path)
    return torch.from_numpy(text_emb_clip_aesthetic)


def average_prompt_embed_with_aesthetic_embed(
    prompt_embed: torch.Tensor,
    aesthetic_embed: torch.Tensor,
    aesthetic_weight: float = 0.5,
) -> torch.Tensor:
    return F.normalize(
        prompt_embed * (1 - aesthetic_weight) + aesthetic_embed * aesthetic_weight
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="inpaint.pt",
        help="path to the diffusion model",
    )

    parser.add_argument(
        "--kl_path",
        type=str,
        default="kl-f8.pt",
        help="path to the LDM first stage model",
    )

    parser.add_argument(
        "--bert_path",
        type=str,
        default="bert.pt",
        help="path to the LDM first stage model",
    )

    parser.add_argument(
        "--text", type=str, required=False, default="", help="your text prompt"
    )

    parser.add_argument(
        "--edit",
        type=str,
        required=False,
        help="path to the image you want to edit (either an image file or .npy containing a numpy array of the image embeddings)",
    )

    parser.add_argument(
        "--mask",
        type=str,
        required=False,
        help="path to a mask image. white pixels = keep, black pixels = discard. width = image width/8, height = image height/8",
    )

    parser.add_argument(
        "--negative", type=str, required=False, default="", help="negative text prompt"
    )

    parser.add_argument(
        "--init_image", type=str, required=False, default=None, help="init image to use"
    )

    parser.add_argument(
        "--skip_timesteps",
        type=int,
        required=False,
        default=0,
        help="how many diffusion steps are gonna be skipped",
    )

    parser.add_argument(
        "--prefix", type=str, required=False, default="", help="prefix for output files"
    )
    parser.add_argument(
        "--num_batches", type=int, default=1, required=False, help="number of batches"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, required=False, help="batch size"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        required=False,
        help="image size of output (multiple of 8)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        required=False,
        help="image size of output (multiple of 8)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=25,
        required=False,
        help="number of mutation steps",
    )
    parser.add_argument(
        "--starting_threshold",
        type=float,
        default=0.6,
        required=False,
        help="how much of the image to replace at the start of editing (1 = inpaint the entire image)",
    )
    parser.add_argument(
        "--ending_threshold",
        type=float,
        default=0.5,
        required=False,
        help="how much of the image to replace at the end of editing",
    )
    parser.add_argument(
        "--starting_radius",
        type=float,
        default=5,
        required=False,
        help="size of noise blur at the start of editing (larger = coarser changes)",
    )
    parser.add_argument(
        "--ending_radius",
        type=float,
        default=0.1,
        required=False,
        help="size of noise blur at the end of editing (smaller = editing fine details)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, required=False, help="random seed"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        required=False,
        help="classifier-free guidance scale",
    )
    parser.add_argument(
        "--steps", type=int, default=0, required=False, help="number of diffusion steps"
    )
    parser.add_argument("--cpu", dest="cpu", action="store_true")
    parser.add_argument("--clip_score", dest="clip_score", action="store_true")
    parser.add_argument("--clip_guidance", dest="clip_guidance", action="store_true")
    parser.add_argument(
        "--clip_guidance_scale",
        type=float,
        default=150,
        required=False,
        help="Controls how much the image should look like the prompt",
    )  # may need to use lower value for ddim
    parser.add_argument(
        "--cutn", type=int, default=16, required=False, help="Number of cuts"
    )
    parser.add_argument(
        "--ddim", dest="ddim", action="store_true", help="turn on to use 50 step ddim"
    )
    parser.add_argument(
        "--ddpm", dest="ddpm", action="store_true", help="turn on to use 50 step ddim"
    )
    parser.add_argument("--aesthetic_rating", type=int, default=9)
    parser.add_argument("--aesthetic_weight", type=float, default=0.5)
    parser.add_argument("--wandb_name", type=str, default="ongo_eval")
    return parser.parse_args()


def fetch(url_or_path):
    if str(url_or_path).startswith("http://") or str(url_or_path).startswith(
        "https://"
    ):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, "rb")


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.0):
        super().__init__()

        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(
                torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size
            )
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), "replicate")
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def load_diffusion_model(model_path, ddpm, ddim, steps, cpu, clip_guidance, device):
    model_state_dict = torch.load(model_path, map_location="cpu")

    model_params = {
        "attention_resolutions": "32,16,8",
        "class_cond": False,
        "diffusion_steps": 1000,
        "rescale_timesteps": True,
        "timestep_respacing": "27",  # Modify this value to decrease the number of
        # timesteps.
        "image_size": 32,
        "learn_sigma": False,
        "noise_schedule": "linear",
        "num_channels": 320,
        "num_heads": 8,
        "num_res_blocks": 2,
        "resblock_updown": False,
        "use_fp16": False,
        "use_scale_shift_norm": False,
        "clip_embed_dim": 768 if "clip_proj.weight" in model_state_dict else None,
        "image_condition": True
        if model_state_dict["input_blocks.0.0.weight"].shape[1] == 8
        else False,
        "super_res_condition": True
        if "external_block.0.0.weight" in model_state_dict
        else False,
    }

    if ddpm:
        model_params["timestep_respacing"] = 1000
    if ddim:
        if steps:
            model_params["timestep_respacing"] = "ddim" + str(steps)
        else:
            model_params["timestep_respacing"] = "ddim50"
    elif steps:
        model_params["timestep_respacing"] = str(steps)

    model_config = model_and_diffusion_defaults()
    model_config.update(model_params)

    if cpu:
        model_config["use_fp16"] = False

    # Load models
    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(model_state_dict, strict=False)
    model.requires_grad_(clip_guidance).eval().to(device)

    if model_config["use_fp16"]:
        model.convert_to_fp16()
    else:
        model.convert_to_fp32()

    return model, model_params, diffusion


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


# vae
def load_vae(clip_guidance: bool, kl_path: str = "kl-f8.pt", device: str = "cuda"):
    ldm = torch.load(kl_path, map_location="cpu")
    ldm.to(device)
    ldm.eval()
    ldm.requires_grad_(clip_guidance)
    set_requires_grad(ldm, clip_guidance)
    return ldm


# bert-text
def load_bert(bert_path: str = "bert.pt", device: str = "cuda"):
    bert = BERTEmbedder(1280, 32)
    sd = torch.load(bert_path, map_location="cpu")
    bert.load_state_dict(sd)

    bert.to(device)
    bert.half().eval()
    set_requires_grad(bert, False)
    return bert


normalize = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
)

# clip
def load_clip_model(device):
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device, jit=False)
    clip_model.eval().requires_grad_(False)
    return clip_model, clip_preprocess


# bert context
def encode_bert(text, negative, batch_size, device, bert=None):
    text_emb = bert.encode([text] * batch_size).to(device).float()
    text_blank = bert.encode([negative] * batch_size).to(device).float()
    return text_emb, text_blank


def encode_clip(clip_model, text, negative, batch_size, device):
    text = clip.tokenize([text] * batch_size, truncate=True).to(device)
    text_clip_blank = clip.tokenize([negative] * batch_size, truncate=True).to(device)
    text_emb_clip = clip_model.encode_text(text)
    text_emb_clip_blank = clip_model.encode_text(text_clip_blank)
    text_emb_norm = text_emb_clip[0] / text_emb_clip[0].norm(dim=-1, keepdim=True)
    return text_emb_clip_blank, text_emb_clip, text_emb_norm


def prepare_edit(ldm, edit, batch_size, width, height, device):
    if edit.endswith(".npy"):
        with open(edit, "rb") as f:
            input_image = np.load(f)
            input_image = torch.from_numpy(input_image).unsqueeze(0).to(device)
            input_image_pil = ldm.decode(input_image)
            input_image_pil = TF.to_pil_image(
                input_image_pil.squeeze(0).add(1).div(2).clamp(0, 1)
            )

            input_image *= 0.18215
    else:
        input_image_pil = Image.open(fetch(edit)).convert("RGB")
        input_image_pil = ImageOps.fit(input_image_pil, (width, height))

        input_image = transforms.ToTensor()(input_image_pil).unsqueeze(0).to(device)
        input_image = 2 * input_image - 1
        input_image = 0.18215 * ldm.encode(input_image).sample()
    image_embed = torch.cat(batch_size * 2 * [input_image], dim=0).float()
    return image_embed


def create_model_fn(model, guidance_scale):
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    return model_fn


def population_path(prefix, i):
    return f"{BASE_DIR}/{prefix}{i:05}.png"


def decode_and_save_image(ldm, prefix, i, image):
    image /= 0.18215
    im = image.unsqueeze(0)
    out = ldm.decode(im)
    npy_filename = f"{BASE_DIR}_npy/{prefix}{i:05}.npy"
    with open(npy_filename, "wb") as outfile:
        np.save(outfile, image.detach().cpu().numpy())

    out = TF.to_pil_image(out.squeeze(0).add(1).div(2).clamp(0, 1))

    filename = population_path(prefix, i)
    out.save(filename)


def pack_model_kwargs(
    text_emb=None,
    text_blank=None,
    text_emb_clip=None,
    text_emb_clip_blank=None,
    image_embed=None,
    model_params=None,
):
    return {
        "context": torch.cat([text_emb, text_blank], dim=0).float(),
        "clip_embed": torch.cat([text_emb_clip, text_emb_clip_blank], dim=0).float()
        if model_params["clip_embed_dim"]
        else None,
        "image_embed": image_embed,
    }


def autoedit(
    i,
    population,
    population_scores,
    model,
    diffusion,
    ldm,
    text_emb_norm,
    clip_model,
    clip_preprocess,
    image_embed,
    model_kwargs,
    batch_size,
    prefix=None,
    device=None,
    ddpm=False,  # TODO
    ddim=False,
    guidance_scale=None,  # TODO
    width=256,
    height=256,
    iterations=30,
    starting_radius=0.6,
    ending_radius=0.1,
    starting_threshold=0.5,
    ending_threshold=0.1,
):
    print("iteration ", i)
    if ddpm:
        sample_fn = diffusion.ddpm_sample_loop_progressive
    elif ddim:
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.plms_sample_loop_progressive

    model_fn = create_model_fn(model, guidance_scale)
    samples = sample_fn(
        model_fn,
        (batch_size * 2, 4, int(height / 8), int(width / 8)),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        cond_fn=None,
        device=device,
        progress=True,
        init_image=None,
        skip_timesteps=0,
    )

    for j, sample in enumerate(samples):
        pass

    for k, image in enumerate(sample["pred_xstart"][:batch_size]):
        im = image / 0.18215
        im = im.unsqueeze(0)
        out = ldm.decode(im)

        out = TF.to_pil_image(out.squeeze(0).add(1).div(2).clamp(0, 1))

        image_emb = clip_model.encode_image(
            clip_preprocess(out).unsqueeze(0).to(device)
        )
        image_emb_norm = image_emb / image_emb.norm(dim=-1, keepdim=True)

        similarity = torch.nn.functional.cosine_similarity(
            image_emb_norm, text_emb_norm, dim=-1
        )

        if i == 0:
            population.append(image.unsqueeze(0))
            population_scores.append(similarity)

            decode_and_save_image(ldm, prefix, k, image.detach().clone())
        elif similarity > population_scores[k]:
            population[k] = image.unsqueeze(0)
            population_scores[k] = similarity
            decode_and_save_image(ldm, prefix, k, image.detach().clone())
            print(k, similarity.item())

    image_embed = torch.cat(population + population, dim=0)
    radius = (starting_radius - ending_radius) * (1 - (i / iterations)) + ending_radius
    blur = transforms.GaussianBlur(kernel_size=(15, 15), sigma=radius)
    mask = torch.randn(batch_size, 1, height // 8, width // 8)
    mask = blur(mask)
    q = (starting_threshold - ending_threshold) * (
        1 - (i / iterations)
    ) + ending_threshold
    threshold = torch.quantile(mask, q)
    mask = (mask > threshold).float()
    # im_mask = TF.to_pil_image(mask[0])
    # im_mask.save('mask_recur.png')
    mask = mask.repeat(1, 4, 1, 1).to(device)
    mask = torch.cat([mask, mask], dim=0)
    image_embed *= mask
    return population, population_scores, image_embed


def autoedit_simulation(
    iterations: int,
    text: str,
    edit: str,
    negative: str,
    prefix: str,
    batch_size: int,
    height: int,
    width: int,
    starting_radius: float,
    ending_radius: float,
    starting_threshold: float,
    ending_threshold: float,
    guidance_scale: float,
    aesthetic_rating: int,
    aesthetic_weight: float,
    model,
    diffusion,
    bert,
    clip_model,
    clip_preprocess,
    ldm,
    model_params,
    device,
):
    # Text Setup
    print(f"Encoding text embeddings with {text} dimensions")
    text_emb, text_blank = encode_bert(text, negative, batch_size, device, bert)
    text_emb_clip_blank, text_emb_clip, text_emb_norm = encode_clip(
        clip_model=clip_model,
        text=text,
        negative=negative,
        batch_size=batch_size,
        device=device,
    )
    print(
        f"Using aesthetic embedding {aesthetic_rating} with weight {aesthetic_weight}"
    )
    text_emb_clip_aesthetic = load_aesthetic_vit_l_14_embed(
        aesthetic_rating, "aesthetic-predictor/vit_l_14_embeddings"
    ).to(device)
    text_emb_clip = average_prompt_embed_with_aesthetic_embed(
        text_emb_clip, text_emb_clip_aesthetic, aesthetic_weight
    )

    # Image Setup
    print(f"Loading image")
    image_embed = None
    if edit:
        image_embed = prepare_edit(ldm, edit, batch_size, width, height, device)
    elif model_params["image_condition"]:
        # using inpaint model but no image is provided
        print(f"Using inpaint model but no image is provided. Initializing with zeros.")
        image_embed = torch.zeros(
            batch_size * 2, 4, height // 8, width // 8, device=device
        )

    # Prepare inputs
    kwargs = pack_model_kwargs(
        text_emb=text_emb,
        text_blank=text_blank,
        text_emb_clip=text_emb_clip,
        text_emb_clip_blank=text_emb_clip_blank,
        image_embed=image_embed,
        model_params=model_params,
    )

    # Initialize population
    population = []
    population_scores = []

    # run simulation
    for population_iteration in range(iterations):
        population, population_scores, image_embed = autoedit(
            population_iteration,
            population,
            population_scores,
            model,
            diffusion,
            ldm,
            text_emb_norm,
            clip_model,
            clip_preprocess,
            image_embed,
            kwargs,
            batch_size,
            prefix,
            device=device,
            ddpm=False,
            ddim=False,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            iterations=iterations,
            starting_radius=starting_radius,
            ending_radius=ending_radius,
            starting_threshold=starting_threshold,
            ending_threshold=ending_threshold,
        )
    return population, population_scores


def main(args):
    """Main function. Runs the model."""

    wandb.init(project=args.wandb_name, config=args)

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    )
    print("Using device:", device)
    if args.seed >= 0:
        torch.manual_seed(args.seed)

    # Model Setup
    print(f"Loading model from {args.model_path}")
    model, model_params, diffusion = load_diffusion_model(
        args.model_path,
        args.ddpm,
        args.ddim,
        args.steps,
        args.cpu,
        args.clip_guidance,
        device,
    )
    print(f"Loading vae")
    ldm = load_vae(
        clip_guidance=args.clip_guidance, kl_path=args.kl_path, device=device
    )
    print(f"Loading CLIP")
    clip_model, clip_preprocess = load_clip_model(device)
    print(f"Loading BERT")
    bert = load_bert(args.bert_path, device)

    if args.text.endswith(".json") and os.path.isfile(args.text):
        texts = json.load(open(args.text))
        print(f"Using text from {args.text}")
    else:
        texts = [args.text]
        print(f"Using text {args.text}")
    eval_table_artifact = wandb.Artifact(
        "glid-3-xl-table" + str(wandb.run.id), type="predictions"
    )
    columns = ["latents", "decoded", "scores"]
    eval_table = wandb.Table(columns=columns)

    for text in texts:
        print(f"Running simulation for {text}")
        # Create new run and table for each prompt.
        prefix = (
            text.replace(" ", "_").replace(",", "_").replace(".", "_").replace("'", "_")
        )
        prefix = prefix[:255]

        population, population_scores = autoedit_simulation(
            iterations=args.iterations,
            text=text,
            edit=args.edit,
            negative=args.negative,
            prefix=prefix,
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
            starting_radius=args.starting_radius,
            ending_radius=args.ending_radius,
            starting_threshold=args.starting_threshold,
            ending_threshold=args.ending_threshold,
            aesthetic_rating=args.aesthetic_rating,
            aesthetic_weight=args.aesthetic_weight,
            guidance_scale=args.guidance_scale,
            model=model,
            diffusion=diffusion,
            bert=bert,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            ldm=ldm,
            model_params=model_params,
            device=device,
        )

        eval_table.add_data(
            [
                wandb.Image(candidate.detach().add(1).div(2).clamp(0, 1))
                for candidate in population
            ],
            [wandb.Image(population_path(prefix, i)) for i in range(len(population))],
            torch.cat(population_scores).detach().cpu().numpy(),
        )
        print(f"Finished simulation for {text}")
        print(f"Final population score: {population_scores}")

    print(f"Generation finished. Syncing table to w&b.")
    eval_table_artifact.add(eval_table, f"{prefix}_eval_table")
    wandb.run.log_artifact(eval_table_artifact)
    wandb.run.finish()


if __name__ == "__main__":
    try:
        main(parse_args())
    except KeyboardInterrupt as kb_interrupt:
        print(f"Keyboard Interrupt. Finishing run.")
        wandb.run.finish()
        raise
