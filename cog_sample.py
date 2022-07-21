import typing

from PIL import Image 

import os

from guided_diffusion.inpaint_util import sample_inpaint, prepare_inpaint_models

import cog
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF

os.environ[
    "TOKENIZERS_PARALLELISM"
] = "false"  # required to avoid errors with transformers lib

class Predictor(cog.BasePredictor):
    @torch.inference_mode()
    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fp16 = True
        self.inpaint_models = prepare_inpaint_models(inpaint_model_path="simulacra_540K.pt", device=self.device, use_fp16=self.use_fp16)


    @torch.inference_mode()
    def predict(
        self,
        prompt: str = cog.Input(description="Your text prompt.", default=""),
        negative: str = cog.Input(
            default="",
            description="(optional) Negate the model's prediction for this text from the model's prediction for the target text.",
        ),
        init_image: cog.Path = cog.Input(
            default=None,
            description="(optional) Initial image to use for the model's prediction. If provided alongside a mask, the image will be inpainted instead.",
        ),
        mask: cog.Path = cog.Input(default=None, description='a mask image for inpainting an init_image. white pixels = keep, black pixels = discard. resized to width = image width/8, height = image height/8'),
        guidance_scale: float = cog.Input(
            default=5.0,
            description="Classifier-free guidance scale. Higher values will result in more guidance toward caption, with diminishing returns. Try values between 1.0 and 40.0. In general, going above 5.0 will introduce some artifacting.",
            le=100.0,
            ge=-20.0,
        ),
        steps: int = cog.Input(
            default=100,
            description="Number of diffusion steps to run. Due to PLMS sampling, using more than 100 steps is unnecessary and may simply produce the exact same output.",
            le=250,
            ge=15,
        ),
        batch_size: int = cog.Input(
            default=1, description="Batch size. (higher = slower)", ge=1, le=16,
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
        init_skip_fraction: float = cog.Input(
            default=0.0,
            description="Fraction of sampling steps to skip when using an init image. Defaults to 0.0 if init_image is not specified and 0.5 if init_image is specified.",
            ge=0.0,
            le=1.0,
        ),
        aesthetic_rating: int = cog.Input(
            description="Aesthetic rating (1-9) - embed to use.", default=9
        ),
        aesthetic_weight: float = cog.Input(
            description="Aesthetic weight (0-1). How much to guide towards the aesthetic embed vs the prompt embed.",
            default=0.5,
        ),
        seed: int = cog.Input(
            default=-1,
            description="Seed for random number generator. If -1, a random seed will be chosen.",
            ge=-1,
            le=(2**32 - 1),
        ),
        intermediate_outputs: bool = cog.Input(
            default=False,
            description="Whether to return intermediate outputs. Enable to visualize the diffusion process and/or debug the model. May slow down inference.",
        ),
    ) -> typing.Iterator[typing.List[cog.Path]]:

        yield from sample_inpaint(
            prompt=prompt,
            negative=negative,
            init_image=str(init_image) if init_image else None,
            mask=str(mask) if mask else None,
            steps=steps,
            init_skip_fraction=init_skip_fraction,
            width=width,
            height=height,
            batch_size=batch_size,
            intermediate_outputs=intermediate_outputs,
            guidance_scale=guidance_scale,
            aesthetic_rating=aesthetic_rating,
            aesthetic_weight=aesthetic_weight,
            device=self.device,
            use_fp16=self.use_fp16,
            seed=seed,
            loaded_models=self.inpaint_models,
        )