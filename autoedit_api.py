import os

os.environ[
    "TOKENIZERS_PARALLELISM"
] = "false"  # required to avoid errors with transformers lib

import sys

sys.path.append("ldm")
import random
import typing

import cog
import torch

from autoedit import (autoedit_simulation, load_bert, load_clip_model,
                      load_diffusion_model, load_vae, autoedit_path)

model_path = "ongo.pt"
kl_path = "kl-f8.pt"
bert_path = "bert.pt"


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

        """Run the model on the given input and return the result"""
        if seed < 0:
            seed = random.randint(0, 2**31)
        print(f"Using seed: {seed}")
        torch.manual_seed(seed)

        print(f"Running autoedit for {text} {negative} {edit}")
        prefix = (
            text.replace(" ", "_").replace(",", "_").replace(".", "_").replace("'", "_")
        )
        prefix = prefix[:255]

        population, population_scores = autoedit_simulation(
            iterations=iterations,
            text=text,
            edit=edit,
            negative=negative,
            prefix=prefix,
            batch_size=batch_size,
            height=height,
            width=width,
            starting_radius=starting_radius,
            ending_radius=ending_radius,
            starting_threshold=starting_threshold,
            ending_threshold=ending_threshold,
            aesthetic_rating=aesthetic_rating,
            aesthetic_weight=aesthetic_weight,
            guidance_scale=guidance_scale,
            model=self.model,
            diffusion=self.diffusion,
            bert=self.bert,
            clip_model=self.clip_model,
            clip_preprocess=self.clip_preprocess,
            ldm=self.ldm,
            model_params=self.model_params,
            device=self.device,
        )
        return [cog.Path(autoedit_path(prefix, i)) for i in range(len(population))]
