import os
import random
import typing
from pathlib import Path
import re
import unicodedata


import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from guided_diffusion.predict_util import (
    average_prompt_embed_with_aesthetic_embed, bert_encode_cfg,
    load_aesthetic_vit_l_14_embed, load_bert,
    load_clip_model_and_transform, load_diffusion_model, load_vae, pack_model_kwargs,
    prepare_edit)
from guided_diffusion.script_util import create_gaussian_diffusion
from dist.clip_custom import clip


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


normalize = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
)

os.environ[
    "TOKENIZERS_PARALLELISM"
] = "false"  # required to avoid errors with transformers lib


KL_PATH = "kl-f8.pt"
BERT_PATH = "bert.pt"


def prepare_inpaint_models(
    inpaint_model_path: str = "inpaint.pt", device: str = "cuda", use_fp16: bool = False
):
    device = torch.device(device)
    print(f"Loading latent diffusion model from {inpaint_model_path}")
    inpaint_model, inpaint_model_config, inpaint_diffusion = load_diffusion_model(
        model_path=inpaint_model_path,
        steps="1000",  # Init method requires steps, although we can modify it during inference as well.
        use_fp16=use_fp16,
        device=device,
    )

    print(f"Loading VAE from {KL_PATH}")
    vae_backbone = load_vae(kl_path=KL_PATH, device=device, use_fp16=use_fp16)

    print(f"Loading CLIP text encoder from textual.onnx")
    clip_model, clip_preprocess = load_clip_model_and_transform(
        device
    )

    print(f"Loading BERT text encoder from {BERT_PATH}")
    bert = load_bert(BERT_PATH, device, use_fp16=use_fp16)
    return dict(
        inpaint_model=inpaint_model,
        inpaint_model_config=inpaint_model_config,
        inpaint_diffusion=inpaint_diffusion,
        vae_backbone=vae_backbone,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        bert=bert,
    )



def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def sample_inpaint(
    prompt: str,
    negative: str = "",
    init_image: str = None,
    mask: str = None,
    steps: int = 100,
    init_skip_fraction: float = 0.5,
    width: int = 256,
    height: int = 256,
    batch_size: int = 1,
    intermediate_outputs: bool = False,
    guidance_scale: float = 0.0,
    aesthetic_rating: int = 9,
    aesthetic_weight: float = 0.0,
    device: str = "cuda",
    use_fp16: bool = False,
    seed: int = 0,
    output_dir: str = "inpaint_outputs",
    loaded_models: typing.Dict = None,
):
    """Predict a normal distribution from a prompt.

    Args:
        prompt: The prompt to use.
        negative: The negative prompt to use.
        init_image: The image to use as the initial image.
        steps: The number of steps to run the model.
        mask: The mask to use for the initial image.
        init_skip_fraction: The fraction of timesteps to skip when using init_image.
        width: The width of the output image.
        height: The height of the output image.
        batch_size: The batch size to use.
        intermediate_outputs: Whether to save intermediate outputs.
        guidance_scale: The scale to use for guidance.
        aesthetic_rating: The rating to use for the aesthetic embedding.
        aesthetic_weight: The weight to use for the aesthetic embedding.
        device: The device to use.
        use_fp16: Whether to use fp16.
        loaded_models: A dictionary of models pre-loaded to `device` to use for inference. Keys are:
            - inpaint_model
            - inpaint_model_config
            - inpaint_diffusion
            - vae_backbone
            - clip_model
            - clip_preprocess
            - bert

    Returns:
        A generator that yields a list of paths to images.
    """
    prompt_dir = Path(output_dir).joinpath(slugify(prompt))
    prompt_dir.mkdir(parents=True, exist_ok=True)
    if seed > 0:
        torch.manual_seed(seed)
    else:
        seed = random.randint(0, 2**32)
        torch.manual_seed(seed)
        print(f"Using seed {seed}")

    if loaded_models is None:
        loaded_models = prepare_inpaint_models(device=device, use_fp16=use_fp16)
    else:
        print("Using preloaded models")

    model_config = loaded_models["inpaint_model_config"]
    clip_model = loaded_models["clip_model"]
    bert = loaded_models["bert"]
    vq_decoder = loaded_models["vae_backbone"]
    inpaint_model = loaded_models["inpaint_model"]

    # Create diffusion manually so we don't re-init the model just to change timestep_respacing
    model_config["timestep_respacing"] = str(steps)
    diffusion = create_gaussian_diffusion(
        steps=model_config["diffusion_steps"],
        learn_sigma=model_config["learn_sigma"],
        noise_schedule=model_config["noise_schedule"],
        use_kl=model_config["use_kl"],
        predict_xstart=model_config["predict_xstart"],
        rescale_timesteps=model_config["rescale_timesteps"],
        timestep_respacing=model_config["timestep_respacing"],
    )

    # Text Setup
    print(f"Encoding text embeddings with {prompt} dimensions")
    text_emb, text_blank = bert_encode_cfg(prompt, negative, batch_size, device, bert)

    text_tokens = clip.tokenize([prompt] * batch_size, truncate=True).to(device)
    negative_tokens = clip.tokenize([negative] * batch_size, truncate=True).to(device)
    text_emb_clip = clip_model.encode_text(text_tokens).to(device).float()
    text_emb_clip_blank = clip_model.encode_text(negative_tokens).to(device).float()
    text_emb_norm = text_emb_clip[0] / text_emb_clip[0].norm(dim=-1, keepdim=True)
    print(
        f"Using aesthetic embedding {aesthetic_rating} with weight {aesthetic_weight}"
    )
    text_emb_clip_aesthetic = load_aesthetic_vit_l_14_embed(rating=aesthetic_rating).to(
        device
    )
    text_emb_clip = average_prompt_embed_with_aesthetic_embed(
        text_emb_clip, text_emb_clip_aesthetic, aesthetic_weight
    )

    # Image Setup

    init = None
    init_skip_fraction = 0.0
    init_skip_timesteps = 0

    image_embed = torch.zeros(batch_size * 2, 4, height // 8, width // 8, device=device)
    if init_image and mask:  # if both are provided, the user is inpainting.
        print(f"Using inpaint model with image: {init_image}")
        image_embed = prepare_edit(
            vq_decoder, str(init_image), width, height, device=device
        )
        mask_image = Image.open(mask).convert("L")
        mask_image = mask_image.resize((width // 8, height // 8), Image.ANTIALIAS)
        mask = transforms.ToTensor()(mask_image).unsqueeze(0).to(device)
        mask1 = mask > 0.5
        mask1 = mask1.float()
        image_embed *= mask1
        image_embed = torch.cat(batch_size * 2 * [image_embed], dim=0)
    elif (
        init_image
    ):  # if just the image is provided, the user wants to use the image as the init image.
        if init_skip_fraction == 0.0:
            print(f"Must specify init_skip_fraction > 0.0 when using init_image.")
            print(f"Overriding init_skip_fraction to 0.5")
            init_skip_fraction = 0.5
        print(
            f"Loading initial image {init_image} with init_skip_fraction: {init_skip_fraction}"
        )
        init = Image.open(init_image).convert("RGB")
        init = init.resize((int(width), int(height)), Image.LANCZOS)
        init = TF.to_tensor(init).to(device).unsqueeze(0).clamp(0, 1)
        if use_fp16:
            init = init.half()
        h = vq_decoder.encode(init * 2 - 1).sample() * 0.18215
        init = torch.cat(batch_size * 2 * [h], dim=0)
        # str to int * float -> float
        init_skip_timesteps = (
            int(model_config["timestep_respacing"]) * init_skip_fraction
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
        model_params=model_config,
    )

    # Create a classifier-free guidance sampling function.
    @torch.cuda.amp.autocast(enabled=use_fp16)
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = inpaint_model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    @torch.cuda.amp.autocast(enabled=use_fp16)
    def save_sample(sample: torch.Tensor) -> typing.List[torch.Tensor]:
        """Save a sample of the model's output."""
        final_outputs = []
        for image in sample["pred_xstart"][:batch_size]:
            image /= 0.18215
            im = image.unsqueeze(0)
            out = vq_decoder.decode(im)
            final_outputs.append(out.squeeze(0).add(1).div(2).clamp(0, 1))
        return final_outputs

    sample_fn = diffusion.plms_sample_loop_progressive
    samples = sample_fn(
        model_fn,
        (batch_size * 2, 4, int(height / 8), int(width / 8)),
        clip_denoised=False,
        model_kwargs=kwargs,
        cond_fn=None,
        device=device,
        progress=True,
        init_image=init,
        skip_timesteps=init_skip_timesteps,
    )

    log_interval = 10
    print("Running diffusion...")
    for timestep_idx, sample in enumerate(samples):
        if (
            timestep_idx % log_interval == 0
            and timestep_idx < diffusion.num_timesteps - 1
            and intermediate_outputs
        ):
            print(f"Timestep {timestep_idx+1} - saving sample/s")
            current_batch = save_sample(sample)
            current_batch_paths = []
            for batch_idx, current_image in enumerate(current_batch):
                current_image_path = prompt_dir.joinpath(
                    f"ts_{timestep_idx}-batch_{batch_idx}.png"
                )
                current_batch_paths.append(current_image_path)
                TF.to_pil_image(current_image).save(current_image_path, optimize=True)
            yield current_batch_paths  # List[str]

    print(f"Saving final sample/s")
    current_batch = save_sample(sample)
    current_batch_paths = []
    for batch_idx, current_image in enumerate(current_batch):
        current_image_path = prompt_dir.joinpath(
            f"ts_{timestep_idx}-batch_{batch_idx}.png"
        )
        current_batch_paths.append(current_image_path)
        TF.to_pil_image(current_image).save(current_image_path, optimize=True)
    yield current_batch_paths
