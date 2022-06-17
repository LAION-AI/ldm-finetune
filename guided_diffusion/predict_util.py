from clip_onnx import clip_onnx
from torchvision.transforms import Compose, Resize, ToTensor, InterpolationMode, CenterCrop, Normalize
import os
import typing
from pathlib import Path

import numpy as np
import torch
from clip_onnx import clip_onnx
from PIL import Image, ImageOps
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

from clip_custom import clip
from encoders.modules import BERTEmbedder
from guided_diffusion import predict_util
from guided_diffusion.script_util import (create_gaussian_diffusion,
                                          create_model_and_diffusion,
                                          model_and_diffusion_defaults)

# load from environment if set, otherwise use "outputs"
BASE_DIR = Path(os.environ.get("BASE_DIR", "outputs"))


@torch.no_grad()
@torch.inference_mode()
def sample_diffusion_model(
    latent_diffusion_model: torch.nn.Module = None,
    kl_model: torch.nn.Module = None,
    diffusion_params: dict = None,
    clip_model: torch.nn.Module = None,
    bert: torch.nn.Module = None,
    text: str = None,
    negative: str = "",
    timestep_respacing: str = "100",
    guidance_scale=5.0,
    device: str = "cuda",
    batch_size: int = 4,
    aesthetic_rating: int = 9,
    aesthetic_weight: float = 0.5,
) -> typing.List[torch.Tensor]:
    diffusion = create_gaussian_diffusion(
        steps=diffusion_params["diffusion_steps"],
        learn_sigma=diffusion_params["learn_sigma"],
        noise_schedule=diffusion_params["noise_schedule"],
        use_kl=diffusion_params["use_kl"],
        predict_xstart=diffusion_params["predict_xstart"],
        rescale_timesteps=diffusion_params["rescale_timesteps"],
        timestep_respacing=timestep_respacing,
    )

    height, width = 256, 256  # TODO get this from the model
    print(f"Running simulation for {text}")
    # Create new run and table for each prompt.
    prefix = (
        text.replace(" ", "_").replace(",", "_").replace(".", "_").replace("'", "_")
    )
    prefix = prefix[:255]

    # Text Setup
    print(f"Encoding text embeddings with {text} dimensions")
    text_emb, text_blank = predict_util.encode_bert(
        text, negative, batch_size, device, bert
    )
    text_emb_clip_blank, text_emb_clip, text_emb_norm = predict_util.encode_clip(
        clip_model=clip_model,
        text_tokens=text,
        negative=negative,
        batch_size=batch_size,
        device=device,
    )
    print(
        f"Using aesthetic embedding {aesthetic_rating} with weight {aesthetic_weight}"
    )
    text_emb_clip_aesthetic = predict_util.load_aesthetic_vit_l_14_embed(
        rating=aesthetic_rating
    ).to(device)
    text_emb_clip = predict_util.average_prompt_embed_with_aesthetic_embed(
        text_emb_clip, text_emb_clip_aesthetic, aesthetic_weight
    )
    image_embed = torch.zeros(batch_size * 2, 4, height // 8, width // 8, device=device)

    # Prepare inputs
    kwargs = predict_util.pack_model_kwargs(
        text_emb=text_emb,
        text_blank=text_blank,
        text_emb_clip=text_emb_clip,
        text_emb_clip_blank=text_emb_clip_blank,
        image_embed=image_embed,
        model_params=diffusion_params,
    )

    def save_sample(sample):
        final_outputs = []
        for image in sample["pred_xstart"][:batch_size]:
            image /= 0.18215
            im = image.unsqueeze(0)
            out = kl_model.decode(im)
            final_outputs.append(out.squeeze(0).add(1).div(2).clamp(0, 1))
        return final_outputs

    sample_fn = diffusion.plms_sample_loop_progressive
    samples = sample_fn(
        create_model_fn(latent_diffusion_model, guidance_scale=guidance_scale),
        (batch_size * 2, 4, int(height / 8), int(width / 8)),
        clip_denoised=False,
        model_kwargs=kwargs,
        cond_fn=None,
        device=device,
        progress=False,
        init_image=None,
        skip_timesteps=0,
    )

    print("Sampling from diffusion model...")
    for j, sample in enumerate(samples):
        pass
    return save_sample(sample)


def load_aesthetic_vit_l_14_embed(
    rating: int = 9, embed_dir: Path = Path("aesthetic_clip_embeds")
) -> torch.Tensor:
    assert rating in range(1, 10), "rating must be in [1, 2, 3, 4, 5, 6, 7, 8, 9]"
    embed_path = embed_dir.joinpath(f"rating{rating}.npy")
    text_emb_clip_aesthetic = np.load(embed_path)
    return torch.from_numpy(text_emb_clip_aesthetic)


def average_prompt_embed_with_aesthetic_embed(
    prompt_embed: torch.Tensor,
    aesthetic_embed: torch.Tensor,
    aesthetic_weight: float = 0.5,
) -> torch.Tensor:
    return F.normalize(
        prompt_embed * (1 - aesthetic_weight) + aesthetic_embed * aesthetic_weight
    )


def load_diffusion_model(model_path: str, steps: int, use_fp16: bool, device: str):
    model_state_dict = torch.load(model_path, map_location="cpu")
    model_params = {
        "attention_resolutions": "32,16,8",
        "class_cond": False,
        "diffusion_steps": 1000,
        "rescale_timesteps": True,
        "timestep_respacing": "",  # Modify this value to decrease the number of
        "image_size": 32,
        "learn_sigma": False,
        "noise_schedule": "linear",
        "num_channels": 320,
        "num_heads": 8,
        "num_res_blocks": 2,
        "resblock_updown": False,
        "use_fp16": use_fp16,
        "use_scale_shift_norm": False,
        "clip_embed_dim": 768,  # if "clip_proj.weight" in model_state_dict else None,
        "image_condition": True
        # if model_state_dict["input_blocks.0.0.weight"].shape[1] == 8 # else False,
        # "super_res_condition": True
        # if "external_block.0.0.weight" in model_state_dict
        # else False,
    }
    model_params["timestep_respacing"] = str(steps)
    model_config = model_and_diffusion_defaults()
    model_config.update(model_params)

    if device == "cpu":
        model_config["use_fp16"] = False

    # Load models
    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(model_state_dict, strict=False)
    if model_config["use_fp16"]:
        model.convert_to_fp16()
    else:
        model.convert_to_fp32()
    model.to(device)
    return model, model_config, diffusion


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


# vae
def load_vae(
    kl_path: Path = Path("kl-f8.pt"), clip_guidance: bool = False, device: str = "cuda"
):
    ldm = torch.load(kl_path, map_location="cpu")
    ldm.to(device)
    ldm.eval()
    ldm.requires_grad_(clip_guidance)
    set_requires_grad(ldm, False)
    return ldm


# bert-text
def load_bert(bert_path: Path = Path("bert.pt"), device: str = "cuda"):
    bert = BERTEmbedder(1280, 32)
    sd = torch.load(bert_path, map_location="cpu")
    bert.load_state_dict(sd)
    bert.to(device)
    bert.half().eval()
    set_requires_grad(bert, False)
    return bert


normalize = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

# clip
def load_clip_model(device, visual_path="visual.onnx", textual_path="textual.onnx"):
    onnx_model = clip_onnx(None)
    onnx_model.load_onnx(visual_path, textual_path, logit_scale=100.0000)
    provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
    onnx_model.start_sessions(providers=[provider])
    return onnx_model, _transform(224)


# bert context
def encode_bert(text, negative, batch_size, device, bert=None):
    text_emb = bert.encode([text] * batch_size).to(device).float()
    text_blank = bert.encode([negative] * batch_size).to(device).float()
    return text_emb, text_blank


# clip context
def encode_cfg_text(clip_model, text, negative, batch_size, device):
    text_tokens = clip.tokenize([text] * batch_size, truncate=True)
    text_tokens = text_tokens.detach().cpu().numpy().astype(np.int64)

    negative_tokens = clip.tokenize([negative] * batch_size, truncate=True)
    negative_tokens = negative_tokens.detach().cpu().numpy().astype(np.int64)

    text_emb_clip = torch.tensor(clip_model.encode_text(text_tokens)).to(device)
    text_emb_clip_blank = torch.tensor(clip_model.encode_text(negative_tokens)).to(device)

    text_emb_norm = text_emb_clip[0] / text_emb_clip[0].norm(dim=-1, keepdim=True)
    return text_emb_clip_blank, text_emb_clip, text_emb_norm


def prepare_edit(ldm, edit, batch_size, width, height, device):
    if edit.endswith(".npy"):
        with open(edit, "rb") as f:
            input_image = np.load(f)
            input_image = torch.from_numpy(input_image).unsqueeze(0).to(device)
            input_image_pil = ldm.decode(input_image)
            input_image_pil = TF.to_pil_image(
                input_image_pil.squeeze(0).add(1).div(2).clamp(0, 1)
            )

            input_image *= 0.18215
    else:
        input_image_pil = Image.open(edit).convert("RGB")
        input_image_pil = ImageOps.fit(input_image_pil, (width, height))
        input_image = transforms.ToTensor()(input_image_pil).unsqueeze(0).to(device)
        input_image = 2 * input_image - 1
        input_image = 0.18215 * ldm.encode(input_image).sample()
    image_embed = torch.cat(batch_size * 2 * [input_image], dim=0).float()
    return image_embed


def create_model_fn(model, guidance_scale):
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    return model_fn


def log_autoedit_sample(
    prefix: str,
    batch_index: int,
    simulation_iter: int,
    vae_embed: torch.Tensor,
    decoded_image: torch.Tensor,
    score: torch.Tensor,
    base_dir: Path,
):
    target_path = base_dir.joinpath(
        f"{prefix}_iter_{simulation_iter:03}_batch_{batch_index:03}_score_{score.item():.3f}.png"
    )

    decoded_image_path = target_path.with_suffix(".png")
    npy_filename = target_path.with_suffix(".npy")
    vae_image_path = target_path.with_suffix(".vae.png")

    vae_embed_visual = vae_embed.squeeze(0).detach().clone().add(1).div(2).clamp(0, 1)
    vae_embed_visual_pil = (
        TF.to_pil_image(vae_embed_visual).resize((256, 256)).convert("RGB")
    )
    vae_embed_visual_pil.save(vae_image_path)

    with open(npy_filename, "wb") as outfile:
        np.save(outfile, vae_embed.detach().cpu().numpy())
    decoded_image_pil = TF.to_pil_image(
        decoded_image.squeeze(0).add(1).div(2).clamp(0, 1)
    )
    decoded_image_pil.save(decoded_image_path)
    return decoded_image_path, vae_image_path, npy_filename, score


def pack_model_kwargs(
    text_emb: torch.Tensor = None,
    text_blank: torch.Tensor = None,
    text_emb_clip: torch.Tensor = None,
    text_emb_clip_blank: torch.Tensor = None,
    image_embed: torch.Tensor = None,
    model_params: dict = None,
):
    return {
        "context": torch.cat([text_emb, text_blank], dim=0).float(),
        "clip_embed": torch.cat([text_emb_clip, text_emb_clip_blank], dim=0).float()
        if model_params["clip_embed_dim"]
        else None,
        "image_embed": image_embed,
    }


class MakeCutouts(torch.nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.0):
        super().__init__()

        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(
                torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size
            )
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), "replicate")
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])
