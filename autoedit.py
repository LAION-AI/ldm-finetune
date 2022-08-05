import argparse
import datetime
import json
import os
import typing
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

import wandb
from dist.clip_custom import clip
from guided_diffusion.predict_util import (
    average_prompt_embed_with_aesthetic_embed, bert_encode_cfg, create_cfg_fn,
    load_aesthetic_vit_l_14_embed, load_bert, load_clip_model_and_transform,
    load_diffusion_model, load_vae, log_autoedit_sample, pack_model_kwargs,
    prepare_edit)
from guided_diffusion.respace import SpacedDiffusion

OUTPUT_DIR = "autoedit_outputs_" + datetime.datetime.now().strftime("%d%H%M%S")
assert not os.path.exists(
    OUTPUT_DIR
), f"Output directory {OUTPUT_DIR} already exists. Please renmae or delete it."
os.makedirs(OUTPUT_DIR, exist_ok=False)


def autoedit(
    model: torch.nn.Module,
    diffusion: SpacedDiffusion,
    ldm: torch.nn.Module,
    text_emb_norm: torch.Tensor,
    clip_model: torch.nn.Module,
    clip_preprocess: typing.Callable,
    model_kwargs: dict,
    batch_size: int,
    prefix: str = None,
    device: str = None,
    guidance_scale: float = None,  # TODO
    width: int = 256,
    height: int = 256,
    num_mutations: int = 30,
    starting_radius: float = 0.6,
    ending_radius: float = 0.1,
    starting_threshold: float = 0.5,
    ending_threshold: float = 0.1,
):
    @lru_cache(maxsize=None)
    def init_vae_sample(vae_embed_image):
        vae_embed = vae_embed_image / 0.18215
        return vae_embed.unsqueeze(0)

    @lru_cache(maxsize=None)
    def decode_vae_sample(embed):
        return ldm.decode(embed)

    @lru_cache(maxsize=None)
    def clip_similarity(clip_image_embed, text_emb_norm):
        # The CLIP embedding is needed for image-image similarity used by autoedit
        decoded_image_as_pil = TF.to_pil_image(
            decoded_image.squeeze(0).add(1).div(2).clamp(0, 1)
        )
        clip_input = clip_preprocess(decoded_image_as_pil).unsqueeze(0).to(device)
        clip_image_embed = clip_model.encode_image(clip_input).to(device)
        image_emb_norm = clip_image_embed / clip_image_embed.norm(dim=-1, keepdim=True)
        return torch.nn.functional.cosine_similarity(
            image_emb_norm, text_emb_norm, dim=-1
        )

    population = []
    population_scores = []
    for mutation_idx in range(num_mutations):
        sample_fn = diffusion.plms_sample_loop_progressive
        model_fn = create_cfg_fn(model, guidance_scale)
        samples_gn = sample_fn(
            model_fn,
            (batch_size * 2, 4, int(height / 8), int(width / 8)),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            cond_fn=None,
            device=device,
            progress=False,
            init_image=None,
            skip_timesteps=0,
        )
        for timestep_idx, sample in enumerate(samples_gn):
            pass  # this runs the entire sample generator

        result_batch = []
        improved_result_batch = []

        for batch_idx, current_vae_tensor in enumerate(
            sample["pred_xstart"][:batch_size]
        ):
            # kl-f8 vqgan embedding needs to be divided by 0.18215 to get the correct range
            normalized_vae_image_embed = init_vae_sample(current_vae_tensor)
            decoded_image = decode_vae_sample(normalized_vae_image_embed)
            similarity = clip_similarity(normalized_vae_image_embed, text_emb_norm)
            if mutation_idx == 0:
                population.append(current_vae_tensor.unsqueeze(0))
                population_scores.append(similarity)
            elif similarity > population_scores[batch_idx]:
                improved_result_batch.append(current_vae_tensor.unsqueeze(0))
                population[batch_idx] = current_vae_tensor.unsqueeze(0)
                population_scores[batch_idx] = similarity

            decoded_image_path, npy_filename = log_autoedit_sample(
                prefix=prefix,
                batch_index=batch_idx,
                simulation_iter=mutation_idx,
                decoded_image=decoded_image,
                score=similarity,
                base_dir=Path(OUTPUT_DIR),
            )

            result_batch.append((decoded_image_path, npy_filename, similarity))
        if len(improved_result_batch) > 0:
            tqdm.write(
                f"Population #{mutation_idx} improved CLIP score for {len(improved_result_batch)}/{batch_size} images"
            )
        else:
            tqdm.write(f"Population #{mutation_idx} did not improve")
        yield result_batch

        # begin next mutation
        image_embed = torch.cat(population + population, dim=0)
        radius = (starting_radius - ending_radius) * (
            1 - (mutation_idx / num_mutations)
        ) + ending_radius
        blur = transforms.GaussianBlur(kernel_size=(15, 15), sigma=radius)
        mask = torch.randn(batch_size, 1, height // 8, width // 8)
        mask = blur(mask)
        q = (starting_threshold - ending_threshold) * (
            1 - (mutation_idx / num_mutations)
        ) + ending_threshold
        threshold = torch.quantile(mask, q)
        mask = (mask > threshold).float()  # TODO
        mask = mask.repeat(1, 4, 1, 1).to(device)
        mask = torch.cat([mask, mask], dim=0)
        image_embed *= mask


@torch.cuda.amp.autocast()
@torch.no_grad()
def main(args):
    """Main function. Runs the model."""

    use_wandb = args.wandb_name is not None

    if use_wandb:
        wandb.init(project=args.wandb_name, config=args)
        wandb.config.update(args)
        eval_table_artifact = wandb.Artifact(
            args.wandb_name + "_autoedit", type="predictions"
        )
        columns = [
            "mutation_index",
            "batch_idx",
            "decoded_image_path",
            "similarity",
        ]
        eval_table = wandb.Table(columns=columns)
    else:
        print("Wandb disabled. Specify --wandb_name to use wandb.")

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    )
    print("Using device:", device)
    if args.seed >= 0:
        torch.manual_seed(args.seed)

    # Model Setup
    print(f"Loading model from {args.model_path}")
    model, model_params, diffusion = load_diffusion_model(
        model_path=args.model_path,
        steps=args.steps,
        use_fp16=True,
        device=device,
    )
    print(f"Loading vae")
    ldm = load_vae(kl_path=args.kl_path, device=device, use_fp16=True)
    print(f"Loading CLIP")
    clip_model, clip_preprocess = load_clip_model_and_transform(device=device)
    print(f"Loading BERT")
    bert = load_bert(args.bert_path, device, use_fp16=True)

    if args.text.endswith(".json") and Path(args.text).exists():
        texts = json.load(open(args.text, "r", encoding="utf-8"))
        print(f"Using text from {args.text}")
    else:
        texts = [args.text]
        print(f"Using text {args.text}")

    for text in texts:
        print(f"Running simulation for {text}")
        # Create new run and table for each prompt.
        prefix = (
            text.replace(" ", "_").replace(",", "_").replace(".", "_").replace("'", "_")
        )
        prefix = prefix[:255]

        # Text Setup
        print(f"Encoding text embeddings with {text} dimensions")
        text_emb, text_blank = bert_encode_cfg(
            text, args.negative, args.batch_size, device, bert
        )
        text_tokens = clip.tokenize([text] * args.batch_size, truncate=True).to(device)
        negative_tokens = clip.tokenize([args.negative] * args.batch_size, truncate=True).to(device)
        text_emb_clip = clip_model.encode_text(text_tokens).to(device).float()
        text_emb_clip_blank = clip_model.encode_text(negative_tokens).to(device).float()
        text_emb_norm = text_emb_clip[0] / text_emb_clip[0].norm(dim=-1, keepdim=True)
        print(
            f"Using aesthetic embedding {args.aesthetic_rating} with weight {args.aesthetic_weight}"
        )
        text_emb_clip_aesthetic = load_aesthetic_vit_l_14_embed(
            rating=args.aesthetic_rating
        ).to(device)
        text_emb_clip = average_prompt_embed_with_aesthetic_embed(
            text_emb_clip, text_emb_clip_aesthetic, args.aesthetic_weight
        )
        # Image Setup
        print("Loading image")
        image_embed = None
        if args.edit:
            image_embed = prepare_edit(
                ldm, args.edit, args.batch_size, args.width, args.height, device
            )
        elif model_params["image_condition"]:
            print(
                "Using inpaint model but no image is provided. Initializing with zeros."
            )
            image_embed = torch.zeros(
                args.batch_size * 2, 4, args.height // 8, args.width // 8, device=device
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
        progress_bar = tqdm(
            enumerate(
                autoedit(
                    model=model,
                    diffusion=diffusion,
                    ldm=ldm,
                    text_emb_norm=text_emb_norm,
                    clip_model=clip_model,
                    clip_preprocess=clip_preprocess,
                    model_kwargs=kwargs,
                    batch_size=args.batch_size,
                    prefix=prefix,
                    device=device,
                    guidance_scale=args.guidance_scale,
                    width=args.width,
                    height=args.height,
                    num_mutations=args.iterations,
                    starting_radius=args.starting_radius,
                    ending_radius=args.ending_radius,
                    starting_threshold=args.starting_threshold,
                    ending_threshold=args.ending_threshold,
                )
            ),
            total=args.iterations,
        )

        for mutation_idx, results in progress_bar:
            for batch_idx, (decoded_image_path, npy_filename, similarity) in enumerate(
                results
            ):
                if use_wandb:
                    eval_table.add_data(
                        mutation_idx,
                        batch_idx,
                        wandb.Image(str(decoded_image_path)),
                        similarity,
                    )
        print(f"Finished simulation for {text}")
    if use_wandb:
        print("Finished all texts. Syncing table to w&b.")
        eval_table_artifact.add(eval_table, f"{prefix}_eval_table")
        wandb.run.log_artifact(eval_table_artifact)
        wandb.run.finish()


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
    parser.add_argument("--aesthetic_rating", type=int, default=9)
    parser.add_argument("--aesthetic_weight", type=float, default=0.0)
    parser.add_argument("--wandb_name", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except KeyboardInterrupt as kb_interrupt:
        print("Keyboard Interrupt. Finishing run.")
        if args.wandb_name is not None:
            wandb.run.finish()
