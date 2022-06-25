import random
import typing

from PIL import Image, ImageFile

ImageFile.MAXBLOCK = 2**20

import os

import cog
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm

from guided_diffusion.predict_util import (
    average_prompt_embed_with_aesthetic_embed, bert_encode_cfg, clip_encode_cfg, load_aesthetic_vit_l_14_embed, load_bert, load_clip_model, load_diffusion_model, load_vae, pack_model_kwargs, prepare_edit)
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


MODEL_PATH = "gary.pt"  # Change to e.g. erlich.pt to use a different checkpoint.
assert os.path.exists(MODEL_PATH), f"{MODEL_PATH} not found"


KL_PATH = "kl-f8.pt"
BERT_PATH = "bert.pt"

class Predictor(cog.BasePredictor):
    @torch.inference_mode()
    def setup(self):
        self.device = torch.device("cuda")
        print(f"Loading model from {MODEL_PATH}")
        self.model, self.model_config, self.diffusion = load_diffusion_model(
            model_path=MODEL_PATH,
            steps="100", # Stubbed out for now.
            use_fp16=False,
            device=self.device,
        )
        print(f"Loading vae")
        self.ldm = load_vae(kl_path=KL_PATH, device=self.device)
        print(f"Loading CLIP")
        self.clip_model, self.clip_preprocess = load_clip_model(self.device)
        print(f"Loading BERT")
        self.bert = load_bert(BERT_PATH, self.device)


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
            description="(optional) Initial image to use for the model's prediction.",
        ),
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
            default=1, description="Batch size.", choices=[1, 3, 6, 9, 12, 16]
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
        image_embed = None
        if init_image:
            init_image = str(init_image)
            image_embed = prepare_edit(
                self.ldm, init_image, batch_size, width, height, self.device
            )
            print("Image embedding shape:", image_embed.shape)
        elif self.model_config["image_condition"]:
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
            model_params=self.model_config,
        )

        # Create a classifier-free guidance sampling function
        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        def save_sample(sample: torch.Tensor) -> typing.List[torch.Tensor]:
            """Save a sample of the model's output."""
            final_outputs = []
            for image in sample["pred_xstart"][:batch_size]:
                image /= 0.18215
                im = image.unsqueeze(0)
                out = self.ldm.decode(im)
                final_outputs.append(out.squeeze(0).add(1).div(2).clamp(0, 1))
            return final_outputs

        if init_image:
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
            h = self.ldm.encode(init * 2 - 1).sample() * 0.18215
            init = torch.cat(batch_size * 2 * [h], dim=0)
            # str to int * float -> float
            init_skip_timesteps = (
                int(self.model_config["timestep_respacing"]) * init_skip_fraction
            )
            # float to int
            init_skip_timesteps = int(init_skip_timesteps)
        else:
            init = None
            init_skip_fraction = 0.0
            init_skip_timesteps = 0

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
        for timestep_idx, sample in tqdm(enumerate(samples)):
            if timestep_idx % log_interval == 0 and timestep_idx < self.diffusion.num_timesteps - 1:
                print(f"Timestep {timestep_idx} - saving sample")
                current_batch = save_sample(sample)
                current_batch_paths = []
                for batch_idx, current_image in enumerate(current_batch):
                    current_image_path = f"current_{batch_idx}.jpg"
                    current_batch_paths.append(cog.Path(current_image_path))
                    TF.to_pil_image(current_image).save(current_image_path)
                yield current_batch_paths # List[cog.Path]
            elif timestep_idx == self.diffusion.num_timesteps - 1:
                print(f"Timestep {timestep_idx} - saving final sample")
                current_batch = save_sample(sample)
                current_batch_paths = []
                for batch_idx, current_image in enumerate(current_batch):
                    current_image_path = f"current_{batch_idx}.jpg"
                    current_batch_paths.append(cog.Path(current_image_path))
                    TF.to_pil_image(current_image).save(current_image_path)
                yield current_batch_paths
