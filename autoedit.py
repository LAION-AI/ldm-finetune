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
import os

import datetime

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
    population = []
    population_scores = []
    for mutation_index in range(num_mutations):
        sample_fn = diffusion.plms_sample_loop_progressive
        model_fn = predict_util.create_model_fn(model, guidance_scale)
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
        for timestep_idx, sample in enumerate(samples_gn):
            pass  # this runs the entire sample generator

        for batch_idx, image in enumerate(sample["pred_xstart"][: args.batch_size]):
            # kl-f8 vqgan embedding needs to be divided by 0.18215 to get the correct range
            vae_embed = image / 0.18215
            vae_embed = vae_embed.unsqueeze(0)

            # to get the actual image from the embedding, we decode the embedding
            decoded_image = ldm.decode(vae_embed)

            # The CLIP embedding is needed for image-image similarity used by autoedit
            decoded_image_as_pil = TF.to_pil_image(
                decoded_image.squeeze(0).add(1).div(2).clamp(0, 1)
            )
            image_emb = clip_model.encode_image(
                clip_preprocess(decoded_image_as_pil).unsqueeze(0).to(device)
            )
            image_emb_norm = image_emb / image_emb.norm(dim=-1, keepdim=True)

            similarity = torch.nn.functional.cosine_similarity(
                image_emb_norm, text_emb_norm, dim=-1
            )

            if mutation_index == 0:
                population.append(image.unsqueeze(0))
                population_scores.append(similarity)
                yield None  # dummy yield to run full loop.
            elif similarity > population_scores[batch_idx]:
                population[batch_idx] = image.unsqueeze(0)
                population_scores[batch_idx] = similarity
                print(batch_idx, similarity.item(), "Success! saving.")
                yield predict_util.log_autoedit_sample(
                    prefix=prefix,
                    batch_index=batch_idx,
                    simulation_iter=mutation_index,
                    vae_embed=vae_embed,
                    decoded_image=decoded_image,
                    score=similarity,
                    base_dir=Path(OUTPUT_DIR),
                )

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


@torch.inference_mode()
@torch.no_grad()
@torch.cuda.amp.autocast()
def main(args):
    """Main function. Runs the model."""

    use_wandb = args.wandb_name is not None

    if use_wandb:
        wandb.init(project=args.wandb_name, config=args)
        wandb.config.update(args)
        eval_table_artifact = wandb.Artifact(
            args.wandb_name + "_autoedit", type="predictions"
        )
        columns = ["mutation_index", "text", "vae_embed", "image"]
        eval_table = wandb.Table(columns=columns)
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

        for mutation_index, mutation_paths in enumerate(
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
        ):
            if (
                mutation_paths is not None
            ):  # if it is, the population did worse per CLIP.
                decoded_image_path, vae_image_path, npy_filename, score = mutation_paths
                print(
                    f"Saving mutation index: {mutation_index} | Score: {score} | Image: {decoded_image_path}"
                )
                if use_wandb:
                    eval_table.add_data(
                        mutation_index,
                        text,
                        wandb.Image(str(vae_image_path)),
                        wandb.Image(str(decoded_image_path)),
                        score.item(),
                    )
        print(f"Finished simulation for {text}")
    if use_wandb:
        print(f"Finished all texts. Syncing table to w&b.")
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
        print(f"Keyboard Interrupt. Finishing run.")

    if args.wandb_name is not None:
        wandb.run.finish()
