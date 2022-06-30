import copy
from pydoc import describe
import random
import typing

from PIL import Image 

import os

import cog
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF

from guided_diffusion.predict_util import (
    average_prompt_embed_with_aesthetic_embed,
    bert_encode_cfg,
    clip_encode_cfg,
    load_aesthetic_vit_l_14_embed,
    load_bert,
    load_clip_model,
    load_diffusion_model,
    load_vae,
    pack_model_kwargs,
    prepare_edit,
)
from guided_diffusion.script_util import create_gaussian_diffusion


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


normalize = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
)

os.environ[
    "TOKENIZERS_PARALLELISM"
] = "false"  # required to avoid errors with transformers lib


MODEL_PATH = "erlich_fp16.pt"  # Change to e.g. erlich.pt to use a different checkpoint.
# MODEL_PATH = "ongo_fp16.pt"
# MODEL_PATH = "puck_fp16.pt"
assert os.path.exists(MODEL_PATH), f"{MODEL_PATH} not found"


KL_PATH = "kl-f8.pt"
BERT_PATH = "bert.pt"


class Predictor(cog.BasePredictor):
    @torch.inference_mode()
    def setup(self):
        self.device = torch.device("cuda")
        self.use_fp16 = True

        print(f"Loading latent diffusion model from {MODEL_PATH}")
        self.model, self.model_config, self.diffusion = load_diffusion_model(
            model_path=MODEL_PATH,
            steps="100",  # Init method requires steps, although we can modify it during inference as well.
            use_fp16=self.use_fp16,
            device=self.device,
        )

        print(f"Loading VAE from {KL_PATH}")
        self.vae_backbone = load_vae(kl_path=KL_PATH, device=self.device, use_fp16=self.use_fp16)

        print(f"Loading CLIP text encoder from textual.onnx")
        self.clip_model, self.clip_preprocess = load_clip_model(
            self.device, visual_path=None, textual_path="textual.onnx"
        )

        print(f"Loading BERT text encoder from {BERT_PATH}")
        self.bert = load_bert(BERT_PATH, self.device, use_fp16=self.use_fp16)


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
        if seed > 0:
            torch.manual_seed(seed)
        else:
            seed = random.randint(0, 2**32)
            torch.manual_seed(seed)
            print(f"Using seed {seed}")
        print(f"Running simulation for {prompt}")

        # Create diffusion manually so we don't re-init the model just to change timestep_respacing
        self.model_config["timestep_respacing"] = str(steps)
        self.diffusion = create_gaussian_diffusion(
            steps=self.model_config["diffusion_steps"],
            learn_sigma=self.model_config["learn_sigma"],
            noise_schedule=self.model_config["noise_schedule"],
            use_kl=self.model_config["use_kl"],
            predict_xstart=self.model_config["predict_xstart"],
            rescale_timesteps=self.model_config["rescale_timesteps"],
            timestep_respacing=self.model_config["timestep_respacing"],
        )

        # Text Setup
        print(f"Encoding text embeddings with {prompt} dimensions")
        text_emb, text_blank = bert_encode_cfg(
            prompt, negative, batch_size, self.device, self.bert
        )
        text_emb_clip_blank, text_emb_clip, text_emb_norm = clip_encode_cfg(
            clip_model=self.clip_model,
            text=prompt,
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

        init = None
        init_skip_fraction = 0.0
        init_skip_timesteps = 0

        image_embed = torch.zeros(
            batch_size * 2, 4, height // 8, width // 8, device=self.device
        )
        if init_image and mask: # if both are provided, the user is inpainting.
            print(f"Using inpaint model with image: {init_image}")
            image_embed = prepare_edit(self.vae_backbone, str(init_image), width, height, device=self.device)
            mask_image = Image.open(mask).convert('L')
            mask_image = mask_image.resize((width//8, height//8), Image.ANTIALIAS)
            mask = transforms.ToTensor()(mask_image).unsqueeze(0).to(self.device)
            mask1 = (mask > 0.5)
            mask1 = mask1.float()
            image_embed *= mask1
            image_embed = torch.cat(batch_size*2*[image_embed], dim=0)
        elif init_image: # if just the image is provided, the user wants to use the image as the init image.
            if init_skip_fraction == 0.0:
                print(f"Must specify init_skip_fraction > 0.0 when using init_image.")
                print(f"Overriding init_skip_fraction to 0.5")
                init_skip_fraction = 0.5
            print(
                f"Loading initial image {init_image} with init_skip_fraction: {init_skip_fraction}"
            )
            init = Image.open(init_image).convert("RGB")
            init = init.resize((int(width), int(height)), Image.LANCZOS)
            init = TF.to_tensor(init).to(self.device).unsqueeze(0).clamp(0, 1)
            if self.use_fp16:
                init = init.half()
            h = self.vae_backbone.encode(init * 2 - 1).sample() * 0.18215
            init = torch.cat(batch_size * 2 * [h], dim=0)
            # str to int * float -> float
            init_skip_timesteps = (
                int(self.model_config["timestep_respacing"]) * init_skip_fraction
            )
            # float to int
            init_skip_timesteps = int(init_skip_timesteps)


        # Prepare inputs
        kwargs = pack_model_kwargs(
            text_emb=text_emb,
            text_blank=text_blank,
            text_emb_clip=text_emb_clip,
            text_emb_clip_blank=text_emb_clip_blank,
            image_embed=image_embed,
            model_params=self.model_config,
        )

        # Create a classifier-free guidance sampling function.
        @torch.cuda.amp.autocast(enabled=self.use_fp16)
        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        @torch.cuda.amp.autocast(enabled=self.use_fp16)
        def save_sample(sample: torch.Tensor) -> typing.List[torch.Tensor]:
            """Save a sample of the model's output."""
            final_outputs = []
            for image in sample["pred_xstart"][:batch_size]:
                image /= 0.18215
                im = image.unsqueeze(0)
                out = self.vae_backbone.decode(im)
                final_outputs.append(out.squeeze(0).add(1).div(2).clamp(0, 1))
            return final_outputs

        sample_fn = self.diffusion.plms_sample_loop_progressive
        samples = sample_fn(
            model_fn,
            (batch_size * 2, 4, int(height / 8), int(width / 8)),
            clip_denoised=False,
            model_kwargs=kwargs,
            cond_fn=None,
            device=self.device,
            progress=True,
            init_image=init,
            skip_timesteps=init_skip_timesteps,
        )

        log_interval = 10
        print("Running diffusion...")
        for timestep_idx, sample in enumerate(samples):
            if (
                timestep_idx % log_interval == 0
                and timestep_idx < self.diffusion.num_timesteps - 1
                and intermediate_outputs
            ):
                print(f"Timestep {timestep_idx+1} - saving sample/s")
                current_batch = save_sample(sample)
                current_batch_paths = []
                for batch_idx, current_image in enumerate(current_batch):
                    current_image_path = f"current_{batch_idx}.png"
                    current_batch_paths.append(cog.Path(current_image_path))
                    TF.to_pil_image(current_image).save(current_image_path, optimize=True)
                yield current_batch_paths  # List[cog.Path]

        print(f"Saving final sample/s")
        current_batch = save_sample(sample)
        current_batch_paths = []
        for batch_idx, current_image in enumerate(current_batch):
            current_image_path = f"current_{batch_idx}.png"
            current_batch_paths.append(cog.Path(current_image_path))
            TF.to_pil_image(current_image).save(current_image_path, optimize=True)
        yield current_batch_paths
