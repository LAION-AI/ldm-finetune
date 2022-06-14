import numpy as np
from pathlib import Path
import typing
import wandb
import argparse
import json

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from guided_diffusion.respace import SpacedDiffusion
from guided_diffusion import predict_util


def automask_transform(
    num_mutations: int,
    population: torch.Tensor,
    batch_size: int,
    mutation_index: int,
    starting_radius: float,
    ending_radius: float,
    starting_threshold: float,
    ending_threshold: float,
    width: int,
    height: int,
    device: str,
):
    image_embed = torch.cat(population + population, dim=0)
    radius = (starting_radius - ending_radius) * (
        1 - (mutation_index / num_mutations)
    ) + ending_radius
    blur = transforms.GaussianBlur(kernel_size=(15, 15), sigma=radius)
    mask = torch.randn(batch_size, 1, height // 8, width // 8)
    mask = blur(mask)
    q = (starting_threshold - ending_threshold) * (
        1 - (mutation_index / num_mutations)
    ) + ending_threshold
    threshold = torch.quantile(mask, q)
    mask = (mask > threshold).float()
    mask = mask.repeat(1, 4, 1, 1).to(device)
    mask = torch.cat([mask, mask], dim=0)
    image_embed *= mask
    return image_embed


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
    ddpm: bool = False,  # TODO
    ddim: bool = False,
    guidance_scale: float = None,  # TODO
    clip_guidance_scale: float = 150,
    width: int = 256,
    height: int = 256,
    num_mutations: int = 30,
    starting_radius: float = 0.6,
    ending_radius: float = 0.1,
    starting_threshold: float = 0.5,
    ending_threshold: float = 0.1,
    log_interval: int = 1,
):
    population = []
    population_scores = []
    if ddpm:
        sample_fn = diffusion.ddpm_sample_loop_progressive
    elif ddim:
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.plms_sample_loop_progressive

    model_fn = predict_util.create_model_fn(model, guidance_scale)
    make_cutouts = predict_util.MakeCutouts(clip_model.visual.input_resolution, args.cutn)

    cur_t = None

    def cond_fn(x, t, context=None, clip_embed=None, image_embed=None):
        with torch.enable_grad():
            x = x[: batch_size].detach().requires_grad_()
            n = x.shape[0]
            my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
            kw = {
                "context": context[: batch_size],
                "clip_embed": clip_embed[: batch_size] if model_kwargs["clip_embed_dim"] else None,
                "image_embed": image_embed[: batch_size] if image_embed is not None else None,
            }
            out = diffusion.p_mean_variance(
                model, x, my_t, clip_denoised=False, **model_kwargs
            )
            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            x_in = out["pred_xstart"] * fac + x * (1 - fac)
            x_in /= 0.18215
            x_img = ldm.decode(x_in)
            clip_in = predict_util.normalize(make_cutouts(x_img.add(1).div(2)))
            clip_embeds = clip_model.encode_image(clip_in).float()
            dists = predict_util.spherical_dist_loss(
                clip_embeds.unsqueeze(1), text_emb_clip.unsqueeze(0)
            )
            dists = dists.view([cutn, n, -1])
            losses = dists.sum(2).mean(0)
            loss = losses.sum() * clip_guidance_scale
            return -torch.autograd.grad(loss, x)[0]

    for mutation_index in range(num_mutations):
        cur_t = diffusion.num_timesteps - 1
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
            cur_t -= 1
            if j % log_interval == 0 and j != diffusion.num_timesteps - 1:
                vae_image_paths = []
                decoded_image_paths = []
                npy_paths = []

                for batch_idx, vae_embed in enumerate(
                    sample["pred_xstart"][:batch_size]
                ):
                    # Create some paths
                    target_path = predict_util.autoedit_path(
                        prefix, mutation_index
                    ).joinpath(f"batch_{batch_idx:05}")

                    decoded_image_path = target_path.with_suffix(".png")
                    vae_image_path = target_path.with_suffix(".vae.png")

                    vae_as_npy_filename = target_path.with_suffix(".npy")
                    with open(vae_as_npy_filename, "wb") as outfile:
                        np.save(outfile, vae_embed.detach().cpu().numpy())
                    npy_paths.append(vae_as_npy_filename)

                    # Visualize the 32x32 embed
                    vae_embed = vae_embed / 0.18215
                    vae_embed = vae_embed.unsqueeze(0)
                    vae_embed_visual = (
                        vae_embed.squeeze(0).detach().clone().add(1).div(2).clamp(0, 1)
                    )
                    vae_embed_visual_pil = TF.to_pil_image(vae_embed_visual).resize(
                        (128, 128)
                    )
                    vae_embed_visual_pil.save(vae_image_path)
                    vae_image_paths.append(vae_image_path)

                    # "Decode" the embed into 256x256 pixels
                    vae_decoded = ldm.decode(vae_decoded)
                    decoded_image_pil = TF.to_pil_image(
                        vae_decoded.squeeze(0).add(1).div(2).clamp(0, 1)
                    )
                    decoded_image_pil.save(decoded_image_path)
                    decoded_image_paths.append(decoded_image_path)

                    # Encode (with CLIP) the current decoded 256x256 (noisy) pixels.
                    clip_image_emb = clip_model.encode_image(
                        clip_preprocess(decoded_image_pil).unsqueeze(0).to(device)
                    )
                    # using the norm lets us use cosine similarity to compare embeddings
                    image_emb_norm = clip_image_emb / clip_image_emb.norm(
                        dim=-1, keepdim=True
                    )
                    similarity = torch.nn.functional.cosine_similarity(
                        image_emb_norm, text_emb_norm, dim=-1
                    )
                    if mutation_index == 0:  # Just started, initialize the population
                        population.append(vae_embed.unsqueeze(0))
                        population_scores.append(similarity)
                    elif similarity > population_scores[batch_idx]:  # Replace the worst
                        population[batch_idx] = vae_embed.unsqueeze(0)
                        population_scores[batch_idx] = similarity
                        print(batch_idx, similarity.item())
                    else:
                        print(f"{batch_idx} {similarity.item()} - not replacing")
                yield mutation_index, decoded_image_paths, vae_image_paths, npy_paths
                model_kwargs["image_embed"] = automask_transform(
                    num_mutations=num_mutations,
                    population=population,
                    batch_size=batch_size,
                    mutation_index=mutation_index,
                    starting_radius=starting_radius,
                    ending_radius=ending_radius,
                    starting_threshold=starting_threshold,
                    ending_threshold=ending_threshold,
                    width=width,
                    height=height,
                    device=device,
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
    model, model_params, diffusion = predict_util.load_diffusion_model(
        model_path=args.model_path,
        ddpm=args.ddpm,
        ddim=args.ddim,
        steps=args.steps,
        clip_guidance=args.clip_guidance,
        use_fp16=args.use_fp16,
        device=device,

    )
    print(f"Loading vae")
    ldm = predict_util.load_vae(kl_path=args.kl_path, device=device)
    print(f"Loading CLIP")
    clip_model, clip_preprocess = predict_util.load_clip_model(device)
    print(f"Loading BERT")
    bert = predict_util.load_bert(args.bert_path, device)

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

        # Text Setup
        print(f"Encoding text embeddings with {text} dimensions")
        text_emb, text_blank = predict_util.encode_bert(
            text, args.negative, args.batch_size, device, bert
        )
        text_emb_clip_blank, text_emb_clip, text_emb_norm = predict_util.encode_clip(
            clip_model=clip_model,
            text=text,
            negative=args.negative,
            batch_size=args.batch_size,
            device=device,
        )
        print(
            f"Using aesthetic embedding {args.aesthetic_rating} with weight {args.aesthetic_weight}"
        )
        text_emb_clip_aesthetic = predict_util.load_aesthetic_vit_l_14_embed(
            rating=args.aesthetic_rating
        ).to(device)
        text_emb_clip = predict_util.average_prompt_embed_with_aesthetic_embed(
            text_emb_clip, text_emb_clip_aesthetic, args.aesthetic_weight
        )
        # Image Setup
        print(f"Loading image")
        image_embed = None
        if args.edit:
            image_embed = predict_util.prepare_edit(
                ldm, args.edit, args.batch_size, args.width, args.height, device
            )
        elif model_params["image_condition"]:
            print(
                f"Using inpaint model but no image is provided. Initializing with zeros."
            )
            image_embed = torch.zeros(
                args.batch_size * 2, 4, args.height // 8, args.width // 8, device=device
            )

        # Prepare inputs
        kwargs = predict_util.pack_model_kwargs(
            text_emb=text_emb,
            text_blank=text_blank,
            text_emb_clip=text_emb_clip,
            text_emb_clip_blank=text_emb_clip_blank,
            image_embed=image_embed,
            model_params=model_params,
        )

        for mutation_index, decoded_image_paths, vae_image_paths, npy_paths in autoedit(
            model=model,
            diffusion=diffusion,
            ldm=ldm,
            text_emb_norm=text_emb_norm,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            image_embed=image_embed,
            model_kwargs=kwargs,
            batch_size=args.batch_size,
            prefix=prefix,
            device=device,
            ddpm=args.ddpm,
            ddim=args.ddim,
            guidance_scale=args.guidance_scale,
            width=args.width,
            height=args.height,
            num_mutations=args.iterations,
            starting_radius=args.starting_radius,
            ending_radius=args.ending_radius,
            starting_threshold=args.starting_threshold,
            ending_threshold=args.ending_threshold,
            log_interval=args.log_interval,
        ):
            if use_wandb:
                eval_table.add_data(
                    [wandb.Image(decoded_image_path) for decoded_image_path in decoded_image_paths],
                    [wandb.Image(vae_image_path) for vae_image_path in vae_image_paths],
                    mutation_index,
                )
            print(
                f"Finished mutation index {mutation_index} of {args.iterations} for {text}"
            )
        if use_wandb:
            print(f"Generation finished. Syncing table to w&b.")
            eval_table_artifact.add(eval_table, f"{prefix}_eval_table")
            wandb.run.log_artifact(eval_table_artifact)
            wandb.run.finish()
        print(f"Finished simulation for {text}")


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
        "--clip_guidance",
        type=bool,
        action="store_true",
    )
    parser.add_argument(
        "--clip_guidance_scale",
        type=int,
        default=150,
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


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except KeyboardInterrupt as kb_interrupt:
        print(f"Keyboard Interrupt. Finishing run.")

    if args.wandb_name is not None:
        wandb.run.finish()
