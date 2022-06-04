import datetime
from pathlib import Path
import wandb
import argparse
import json

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

from clip_custom import clip
from encoders.modules import BERTEmbedder
from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

shortened_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
BASE_DIR = Path(f"outputs/autoedit_{shortened_time}")
BASE_DIR.mkdir(exist_ok=True, parents=True)
print(f"Using base directory {BASE_DIR}")


def load_aesthetic_vit_l_14_embed(
    rating: int = 9, embed_dir: Path = Path("aesthetic_clip_embeds")
) -> torch.Tensor:
    assert rating in range(1, 10), "rating must be in [1, 2, 3, 4, 5, 6, 7, 8, 9]"
    embed_path = embed_dir.joinpath(f"rating{rating}.npy")
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
        type=Path,
        default="inpaint.pt",
        help="path to the diffusion model",
    )
    parser.add_argument(
        "--kl_path",
        type=Path,
        default=Path("kl-f8.pt"),
        help="path to the LDM first stage model",
    )
    parser.add_argument(
        "--bert_path",
        type=Path,
        default=Path("bert.pt"),
        help="path to the LDM first stage model",
    )
    parser.add_argument(
        "--text", type=str, required=False, default="", help="your text prompt"
    )
    parser.add_argument(
        "--edit",
        type=Path,
        required=False,
        help="path to the image you want to edit (either an image file or .npy containing a numpy array of the image embeddings)",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        required=False,
        help="path to a mask image. white pixels = keep, black pixels = discard. width = image width/8, height = image height/8",
    )
    parser.add_argument(
        "--negative", type=str, required=False, default="", help="negative text prompt"
    )
    parser.add_argument(
        "--skip_timesteps",
        type=int,
        required=False,
        default=0,
        help="how many diffusion steps are gonna be skipped",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=False,
        default="autoedit",
        help="prefix for output files",
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
        default=10,
        required=False,
        help="number of mutation steps",
    )
    parser.add_argument(
        "--starting_threshold",
        type=float,
        default=0.9,
        required=False,
        help="how much of the image to replace at the start of editing (1 = inpaint the entire image)",
    )
    parser.add_argument(
        "--ending_threshold",
        type=float,
        default=0.8,
        required=False,
        help="how much of the image to replace at the end of editing",
    )
    parser.add_argument(
        "--starting_radius",
        type=float,
        default=3,
        required=False,
        help="size of noise blur at the start of editing (larger = coarser changes)",
    )
    parser.add_argument(
        "--ending_radius",
        type=float,
        default=1.0,
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
    parser.add_argument(
        "--ddim", dest="ddim", action="store_true", help="turn on to use 50 step ddim"
    )
    parser.add_argument(
        "--ddpm", dest="ddpm", action="store_true", help="turn on to use 50 step ddim"
    )
    parser.add_argument("--aesthetic_rating", type=int, default=9)
    parser.add_argument("--aesthetic_weight", type=float, default=0.0)
    parser.add_argument("--wandb_name", type=str, default=None)
    return parser.parse_args()


def load_diffusion_model(model_path, ddpm, ddim, steps, cpu, device):
    model_state_dict = torch.load(model_path, map_location="cpu")

    model_params = {
        "attention_resolutions": "32,16,8",
        "class_cond": False,
        "diffusion_steps": 1000,
        "rescale_timesteps": True,
        "timestep_respacing": "27",  # Modify this value to decrease the number of
        "image_size": 32,
        "learn_sigma": False,
        "noise_schedule": "linear",
        "num_channels": 320,
        "num_heads": 8,
        "num_res_blocks": 2,
        "resblock_updown": False,
        "use_fp16": True,
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
    model.requires_grad_(False).eval().to(device)

    if model_config["use_fp16"]:
        model.convert_to_fp16()
    else:
        model.convert_to_fp32()

    return model, model_params, diffusion


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


# vae
def load_vae(kl_path: Path = Path("kl-f8.pt"), device: str = "cuda"):
    ldm = torch.load(kl_path, map_location="cpu")
    ldm.to(device)
    ldm.eval()
    ldm.requires_grad_(False)
    set_requires_grad(ldm, False)
    return ldm


# bert-text
def load_bert(bert_path: Path = Path("bert.pt"), device: str = "cuda"):
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


# clip context
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
        input_image_pil = Image.open(edit).convert("RGB")
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


def autoedit_path(prefix, simulation_iter):
    target_dir = BASE_DIR.joinpath(f"prefix_{prefix}", f"simulation_{simulation_iter}")
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def log_autoedit_sample(prefix, batch_index, simulation_iter, vae_embed, decoded_image):
    target_path = autoedit_path(prefix, simulation_iter).joinpath(
        f"batch_{batch_index:05}"
    )

    decoded_image_path = target_path.with_suffix(".png")
    npy_filename = target_path.with_suffix(".npy")
    vae_image_path = target_path.with_suffix(".vae.png")

    vae_embed_visual = vae_embed.squeeze(0).detach().clone().add(1).div(2).clamp(0, 1)
    vae_embed_visual_pil = TF.to_pil_image(vae_embed_visual).resize((256, 256)).convert("RGB")
    vae_embed_visual_pil.save(vae_image_path)

    with open(npy_filename, "wb") as outfile:
        np.save(outfile, vae_embed.detach().cpu().numpy())
    decoded_image_pil = TF.to_pil_image(
        decoded_image.squeeze(0).add(1).div(2).clamp(0, 1)
    )
    decoded_image_pil.save(decoded_image_path)
    return decoded_image_path, vae_image_path, npy_filename


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
    population = []
    population_scores = []
    for simulation_iter in range(iterations):
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

        for j, sample in enumerate(samples):  # TODO what the hell does this do?
            pass

        for sample_batch_index, vae_embed in enumerate(sample["pred_xstart"][:batch_size]):
            decoded_image = vae_embed / 0.18215
            decoded_image = decoded_image.unsqueeze(0)
            decoded_image = ldm.decode(decoded_image)

            decoded_image_pil = TF.to_pil_image(
                decoded_image.squeeze(0).add(1).div(2).clamp(0, 1)
            )
            clip_image_emb = clip_model.encode_image(
                clip_preprocess(decoded_image_pil).unsqueeze(0).to(device)
            )
            # using the norm lets us use cosine similarity to compare embeddings
            image_emb_norm = clip_image_emb / clip_image_emb.norm(dim=-1, keepdim=True)
            similarity = torch.nn.functional.cosine_similarity(
                image_emb_norm, text_emb_norm, dim=-1
            )
            if simulation_iter == 0:
                population.append(vae_embed.unsqueeze(0))
                population_scores.append(similarity)
            elif similarity > population_scores[sample_batch_index]:
                population[sample_batch_index] = vae_embed.unsqueeze(0)
                population_scores[sample_batch_index] = similarity
                print(sample_batch_index, similarity.item())
                log_autoedit_sample(
                    prefix,
                    sample_batch_index,
                    simulation_iter,
                    vae_embed.detach().clone(),
                    decoded_image.detach().clone(),
                )
            else:
                print(f"{sample_batch_index} {similarity.item()} - not replacing")

        image_embed = torch.cat(population + population, dim=0)
        radius = (starting_radius - ending_radius) * (
            1 - (simulation_iter / iterations)
        ) + ending_radius
        blur = transforms.GaussianBlur(kernel_size=(15, 15), sigma=radius)
        mask = torch.randn(batch_size, 1, height // 8, width // 8)
        mask = blur(mask)
        q = (starting_threshold - ending_threshold) * (
            1 - (simulation_iter / iterations)
        ) + ending_threshold
        threshold = torch.quantile(mask, q)
        mask = (mask > threshold).float()
        mask = mask.repeat(1, 4, 1, 1).to(device)
        mask = torch.cat([mask, mask], dim=0)
        image_embed *= mask
        # Run simulation
        yield simulation_iter, population, population_scores


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
    text_emb_clip_aesthetic = load_aesthetic_vit_l_14_embed(rating=aesthetic_rating).to(
        device
    )
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
    yield from autoedit(
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

@torch.inference_mode()
@torch.no_grad()
@torch.cuda.amp.autocast()
def main(args):
    """Main function. Runs the model."""

    use_wandb = args.wandb_name is not None

    if use_wandb:
        wandb.init(project=args.wandb_name, config=args)
        wandb.config.update(args)
    else:
        print(f"Wandb disabled. Specify --wandb_name to use wandb.")

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
        device,
    )
    print(f"Loading vae")
    ldm = load_vae(kl_path=args.kl_path, device=device)
    print(f"Loading CLIP")
    clip_model, clip_preprocess = load_clip_model(device)
    print(f"Loading BERT")
    bert = load_bert(args.bert_path, device)

    if args.text.endswith(".json") and Path(args.text).exists():
        texts = json.load(open(args.text))
        print(f"Using text from {args.text}")
    else:
        texts = [args.text]
        print(f"Using text {args.text}")

    if use_wandb:
        eval_table_artifact = wandb.Artifact(
            "glid-3-xl-table" + str(wandb.run.id), type="predictions"
        )
        columns = [
            "latents",
            "decoded",
            "scores",
            "aesthetic_rating",
            "aesthetic_weight",
            "simulation_iter",
        ]
        eval_table = wandb.Table(columns=columns)

    for text in texts:
        print(f"Running simulation for {text}")
        # Create new run and table for each prompt.
        prefix = (
            text.replace(" ", "_").replace(",", "_").replace(".", "_").replace("'", "_")
        )
        prefix = prefix[:255]

        # def autoedit_simulation(#...
        # yield population, population_scores, image_embed
        for simulation_iter, population, population_scores in autoedit_simulation(
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
        ):
            # columns = ["latents", "decoded", "scores", "aesthetic_rating", "aesthetic_weight", "simulation_iter"]
            if use_wandb:
                eval_table.add_data(
                    [
                        wandb.Image(candidate.detach().add(1).div(2).clamp(0, 1))
                        for candidate in population
                    ],
                    [
                        wandb.Image(autoedit_path(prefix, i))
                        for i in range(len(population))
                    ],
                    torch.cat(population_scores).detach().cpu().numpy(),
                    args.aesthetic_rating,
                    args.aesthetic_weight,
                    simulation_iter,
                )
            print(
                f"Finished simulation iter {simulation_iter} of {args.iterations} for {text}"
            )

        if use_wandb:
            print(f"Generation finished. Syncing table to w&b.")
            eval_table_artifact.add(eval_table, f"{prefix}_eval_table")
            wandb.run.log_artifact(eval_table_artifact)
            wandb.run.finish()
        print(f"Finished simulation for {text}")


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except KeyboardInterrupt as kb_interrupt:
        print(f"Keyboard Interrupt. Finishing run.")

    if args.wandb_name is not None:
        wandb.run.finish()
