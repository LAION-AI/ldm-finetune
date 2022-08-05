import os
from random import randint
from typing import Iterator, List

import cog
import torch

from autoedit import autoedit
from guided_diffusion.predict_util import (
    average_prompt_embed_with_aesthetic_embed, bert_encode_cfg,
    load_aesthetic_vit_l_14_embed, load_bert, load_clip_model_and_transform,
    load_diffusion_model, load_vae, pack_model_kwargs, prepare_edit)

os.environ[
    "TOKENIZERS_PARALLELISM"
] = "false"  # required to avoid errors with transformers lib

MODEL_PATH = "erlich.pt"
KL_PATH = "kl-f8.pt"
BERT_PATH = "bert.pt"


class AutoEditOutput(cog.BaseModel):
    image: cog.Path
    similarity: float


class Predictor(cog.BasePredictor):
    @torch.inference_mode()
    def setup(self):
        self.device = torch.device("cuda")
        print(f"Loading model from {MODEL_PATH}")
        self.model, self.model_params, self.diffusion = load_diffusion_model(
            model_path=MODEL_PATH,
            steps="27",
            use_fp16=False,
            device=self.device,
        )
        print(f"Loading vae")
        self.ldm = load_vae(kl_path=KL_PATH, device=self.device)
        self.ldm = self.ldm
        print(f"Loading CLIP")
        self.clip_model, self.clip_preprocess = load_clip_model_and_transform(self.device)
        print(f"Loading BERT")
        self.bert = load_bert(BERT_PATH, self.device)
        self.bert = self.bert

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
            default=1, description="Batch size.", choices=[1, 2, 3, 4, 6, 8]
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
            default=25,
            description="Number of iterations to run the model for.",
            ge=25,
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
    ) -> Iterator[List[cog.Path]]:
        if seed > 0:
            torch.manual_seed(seed)
        else:
            seed = randint(0, 2**32)
            torch.manual_seed(seed)
            print(f"Using seed {seed}")
        print(f"Running simulation for {text}")
        # Create new run and table for each prompt.
        prefix = (
            text.replace(" ", "_").replace(",", "_").replace(".", "_").replace("'", "_")
        )
        prefix = prefix[:255]

        # Text Setup
        print(f"Encoding text embeddings with {text} dimensions")
        text_emb, text_blank = bert_encode_cfg(
            text, negative, batch_size, self.device, self.bert
        )
        text_emb_clip_blank, text_emb_clip, text_emb_norm = clip_encode_cfg(
            clip_model=self.clip_model,
            text=text,
            negative=negative,
            batch_size=batch_size,
            device=self.device,
        )
        print(
            f"Using aesthetic embedding {aesthetic_rating} with weight {aesthetic_weight}"
        )
        text_emb_clip_aesthetic = load_aesthetic_vit_l_14_embed(
            rating=aesthetic_rating
        ).to(self.device)
        text_emb_clip = average_prompt_embed_with_aesthetic_embed(
            text_emb_clip, text_emb_clip_aesthetic, aesthetic_weight
        )
        # Image Setup
        image_embed = None
        if edit:
            image_embed = prepare_edit(
                self.ldm, edit, batch_size, width, height, self.device
            )
            print("Image embedding shape:", image_embed.shape)
        elif self.model_params["image_condition"]:
            print(
                "Using inpaint model but no image is provided. Initializing with zeros."
            )
            image_embed = torch.zeros(
                batch_size * 2, 4, height // 8, width // 8, device=self.device
            )

        # Prepare inputs
        kwargs = pack_model_kwargs(
            text_emb=text_emb,
            text_blank=text_blank,
            text_emb_clip=text_emb_clip,
            text_emb_clip_blank=text_emb_clip_blank,
            image_embed=image_embed,
            model_params=self.model_params,
        )

        for results in autoedit(
            model=self.model,
            diffusion=self.diffusion,
            ldm=self.ldm,
            text_emb_norm=text_emb_norm,
            clip_model=self.clip_model,
            clip_preprocess=self.clip_preprocess,
            model_kwargs=kwargs,
            batch_size=batch_size,
            prefix=prefix,
            device=self.device,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            num_mutations=iterations,
            starting_radius=starting_radius,
            ending_radius=ending_radius,
            starting_threshold=starting_threshold,
            ending_threshold=ending_threshold,
        ):
            outputs = []
            for result in results:
                decoded_image_path, _, _, similarity = result
                # outputs.append(AutoEditOutput(image=cog.Path(str(decoded_image_path)), similarity=similarity))
                outputs.append(cog.Path(str(decoded_image_path)))
            yield outputs