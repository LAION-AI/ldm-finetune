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
    population = []
    population_scores = []

    sample_fn = diffusion.plms_sample_loop_progressive

    model_fn = predict_util.create_model_fn(model, guidance_scale)

    for mutation_index in range(num_mutations):
        samples_gn = sample_fn(
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
        vae_pil_images = []
        decoded_pil_images = []
        npy_paths = []

        # This will iterate through all timesteps for current mutation/population
        final_sample = list(samples_gn)[-1] 
        for batch_idx, vae_embed in enumerate(
            final_sample["pred_xstart"][:batch_size]
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
            vae_pil_images.append(vae_embed_visual_pil)

            # "Decode" the embed into 256x256 pixels
            vae_decoded = ldm.decode(vae_embed)
            decoded_image_pil = TF.to_pil_image(
                vae_decoded.squeeze(0).add(1).div(2).clamp(0, 1)
            )
            decoded_image_pil.save(decoded_image_path)
            decoded_pil_images.append(decoded_image_pil)

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

        yield mutation_index, decoded_pil_images, vae_pil_images, npy_paths, population_scores
        
    image_embed = torch.cat(population+population, dim=0)

    radius = (starting_radius-ending_radius)*(1 - (mutation_index/num_mutations)) + ending_radius
    blur = transforms.GaussianBlur(kernel_size=(15, 15), sigma=radius)
    mask = torch.randn(batch_size, 1, height//8, width//8)
    mask = blur(mask)
    q = (starting_threshold-ending_threshold)*(1 - (mutation_index/num_mutations)) + ending_threshold
    threshold = torch.quantile(mask, q)
    mask = (mask > threshold).float()
    mask = mask.repeat(1, 4, 1, 1).to(device)
    mask = torch.cat([mask,mask], dim=0)
    image_embed *= mask


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
        steps=args.steps,
        use_fp16=True,
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
            "mutation_index",
            "decoded_images",
            "vae_images",
            "population_scores",
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

        # yield mutation_index, decoded_pil_images, vae_pil_images, npy_paths
        for mutation_index, decoded_pil_images, vae_pil_images, npy_paths, population_scores in autoedit(
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
        ):
            if use_wandb:
                columns = [
                    "mutation_index",
                    "decoded_images",
                    "vae_images",
                    "population_scores",
                ]
                eval_table.add_data(
                    mutation_index,
                    [wandb.Image(img, caption=str(idx)) for idx, img in enumerate(decoded_pil_images)],
                    [wandb.Image(img, caption=str(idx)) for idx, img in enumerate(vae_pil_images)],
                    population_scores,

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
