import os

from guided_diffusion.predict_util import average_prompt_embed_with_aesthetic_embed, encode_bert, encode_clip, load_aesthetic_vit_l_14_embed, pack_model_kwargs, prepare_edit

os.environ[
    "TOKENIZERS_PARALLELISM"
] = "false"  # required to avoid errors with transformers lib

import typing
import cog
import torch

from autoedit import (autoedit, autoedit_simulation, load_bert, load_clip_model,
                      load_diffusion_model, load_vae, autoedit_path)

model_path = "ongo.pt"
kl_path = "kl-f8.pt"
bert_path = "bert.pt"



class ModelOutput(cog.BaseModel):
    mutation: int
    score: typing.Optional[float]
    vae_embed: typing.Optional[cog.Path]
    image: typing.Optional[cog.Path]
    npy_file: typing.Optional[cog.File]



class Predictor(cog.BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

        print(f"Loading latent-diffusion model.")
        self.model, self.model_params, self.diffusion = load_diffusion_model(
            model_path, False, False, 27, False, False, self.device
        )

        print(f"Loading VAE.")
        self.ldm = load_vae(clip_guidance=False, kl_path=kl_path, device=self.device)

        print(f"Loading CLIP.")
        self.clip_model, self.clip_preprocess = load_clip_model(self.device)

        print(f"Loading BERT.")
        self.bert = load_bert(bert_path, self.device)

    @torch.inference_mode()
    def predict(
        self,
        text: str = cog.Input(
            default="",
            description="(optional) Text to use for the model's prediction.",
        ),
        edit: str = cog.Input(
            default="",
            description="path to the image you want to edit",
        ),
        negative: str = cog.Input(
            default="",
            description="(optional) Negate the model's prediction for this text from the model's prediction for the target text.",
        ),
        aesthetic_rating: int = cog.Input(
            description="Number between 0 and 9 representing the aesthetic rating. Will initialize the prompt CLIP embed with the respective aesthetic embed.",
            default=9,
            ge=0,
            le=9,
        ),
        aesthetic_weight: float = cog.Input(
            description="Weight of the aesthetic embedding in the average prompt embedding.",
            default=0.5,
            ge=0,
            le=1,
        ),
        batch_size: int = cog.Input(
            default=4, description="Batch size.", choices=[1, 2, 3, 4, 6, 8]
        ),
        width: int = cog.Input(
            default=256,
            description="Target width",
            choices=[128, 192, 256, 320, 384],
        ),
        height: int = cog.Input(
            default=256,
            description="Target height",
            choices=[128, 192, 256, 320, 384],
        ),
        iterations: int = cog.Input(
            default=1,
            description="Number of iterations to run the model for.",
            ge=1,
        ),
        starting_radius: float = cog.Input(
            default=5.0,
            description="size of noise blur at the start of editing (larger = coarser changes)",
            ge=0.1,
        ),
        ending_radius: float = cog.Input(
            default=0.1,
            description="size of noise blur at the end of editing (smaller = editing fine details)",
            ge=0.1,
            le=5.0,
        ),
        starting_threshold: float = cog.Input(
            default=0.6,
            description="how much of the image to replace at the start of editing (1 = inpaint the entire image)",
            ge=0.05,
            le=1.0,
        ),
        ending_threshold: float = cog.Input(
            default=0.5,
            description="how much of the image to replace at the end of editing",
            ge=0.1,
            le=1.0,
        ),
        guidance_scale: float = cog.Input(
            default=5.0,
            description="Controls how much the image should look like the prompt",
            ge=-10.0,
            le=100.0,
        ),
        seed: int = cog.Input(
            default=-1,
            description="(optional) Seed for the random number generator.",
            ge=-1,
        ),
    ) -> typing.List[cog.Path]:
        device = torch.device(
            "cuda" if (torch.cuda.is_available() and not cpu) else "cpu"
        )
        print("Using device:", device)
        if seed >= 0:
            torch.manual_seed(seed)

        # Model Setup
        print(f"Loading model from {model_path}")
        model, model_params, diffusion = load_diffusion_model(
            model_path=model_path,
            steps=steps,
            use_fp16=True,
            device=device,

        )
        print(f"Loading vae")
        ldm = load_vae(kl_path=kl_path, device=device)
        print(f"Loading CLIP")
        clip_model, clip_preprocess = load_clip_model(device)
        print(f"Loading BERT")
        bert = load_bert(bert_path, device)

        if text.endswith(".json") and Path(text).exists():
            texts = json.load(open(text))
            print(f"Using text from {text}")
        else:
            texts = [text]
            print(f"Using text {text}")

        for text in texts:
            print(f"Running simulation for {text}")
            # Create new run and table for each prompt.
            prefix = (
                text.replace(" ", "_").replace(",", "_").replace(".", "_").replace("'", "_")
            )
            prefix = prefix[:255]

            # Text Setup
            print(f"Encoding text embeddings with {text} dimensions")
            text_emb, text_blank = encode_bert(
                text, negative, batch_size, device, bert
            )
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
                rating=aesthetic_rating
            ).to(device)
            text_emb_clip = average_prompt_embed_with_aesthetic_embed(
                text_emb_clip, text_emb_clip_aesthetic, aesthetic_weight
            )
            # Image Setup
            print(f"Loading image")
            image_embed = None
            if edit:
                image_embed = prepare_edit(
                    ldm, edit, batch_size, width, height, device
                )
            elif model_params["image_condition"]:
                print(
                    f"Using inpaint model but no image is provided. Initializing with zeros."
                )
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

            for mutation_index, mutation_paths in enumerate(
                autoedit(
                    model=model,
                    diffusion=diffusion,
                    ldm=ldm,
                    text_emb_norm=text_emb_norm,
                    clip_model=clip_model,
                    clip_preprocess=clip_preprocess,
                    model_kwargs=kwargs,
                    batch_size=batch_size,
                    prefix=prefix,
                    device=device,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    num_mutations=iterations,
                    starting_radius=starting_radius,
                    ending_radius=ending_radius,
                    starting_threshold=starting_threshold,
                    ending_threshold=ending_threshold,
                )
            ):
                if (
                    mutation_paths is not None
                ):  # if it is, the population did worse per CLIP.
                    decoded_image_path, vae_image_path, npy_filename, score = mutation_paths
                    yield ModelOutput(mutation=mutation_index, score=float(score.item()), vae_embed=vae_image_path, image=decoded_image_path, npy_file=npy_filename)