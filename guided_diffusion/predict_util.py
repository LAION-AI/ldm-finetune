from email.mime import base
import sys
import os
import typing
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, Resize, ToTensor)
from torchvision.transforms import functional as TF

from dist.clip_onnx import clip_onnx
from dist.clip_custom import clip
from encoders.modules import BERTEmbedder
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
    """
    Sample a diffusion model.
    """
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

    # BERT Text Setup
    text_emb, text_blank = bert_encode_cfg(text, negative, batch_size, device, bert)

    # CLIP Text Setup
    clip_text_tokens = clip.tokenize([text]*batch_size, truncate=True).to(device)
    clip_blank_tokens = clip.tokenize([negative]*batch_size, truncate=True).to(device)

    clip_text_embed = clip_model.encode_text(clip_text_tokens)
    clip_blank_embed = clip_model.encode_text(clip_blank_tokens)
    clip_text_emb_norm = clip_text_embed[0] / clip_text_embed[0].norm(dim=-1, keepdim=True)

    print(
        f"Using aesthetic embedding {aesthetic_rating} with weight {aesthetic_weight}"
    )
    text_emb_clip_aesthetic = load_aesthetic_vit_l_14_embed(rating=aesthetic_rating).to(
        device
    )
    clip_text_embed = average_prompt_embed_with_aesthetic_embed(
        clip_text_embed, text_emb_clip_aesthetic, aesthetic_weight
    )
    image_embed = torch.zeros(batch_size * 2, 4, height // 8, width // 8, device=device)

    # Prepare inputs
    kwargs = pack_model_kwargs(
        text_emb=text_emb,
        text_blank=text_blank,
        text_emb_clip=clip_text_embed,
        text_emb_clip_blank=clip_blank_embed,
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
        create_cfg_fn(latent_diffusion_model, guidance_scale=guidance_scale),
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
    final_sample = list(samples)[-1]
    return save_sample(final_sample)


def load_aesthetic_vit_l_14_embed(
    rating: int = 9, embed_dir: Path = Path("aesthetic_clip_embeds")
) -> torch.Tensor:
    """
    Load the aesthetic CLIP embedding for the given rating.
    """
    assert rating in range(1, 10), "rating must be in [1, 2, 3, 4, 5, 6, 7, 8, 9]"
    embed_path = embed_dir.joinpath(f"rating{rating}.npy")
    text_emb_clip_aesthetic = np.load(embed_path)
    return torch.from_numpy(text_emb_clip_aesthetic)


def average_prompt_embed_with_aesthetic_embed(
    prompt_embed: torch.Tensor,
    aesthetic_embed: torch.Tensor,
    aesthetic_weight: float = 0.5,
) -> torch.Tensor:
    """
    Average and normalize the prompt embedding with the aesthetic embedding you pass in.
    """
    return F.normalize(
        prompt_embed * (1 - aesthetic_weight) + aesthetic_embed * aesthetic_weight
    )


def load_diffusion_model(model_path: str, steps: int, use_fp16: bool, device: str):
    """
    Load a diffusion model from a checkpoint.
    """
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
    """
    Set the requires_grad flag of all parameters in the model.
    """
    for param in model.parameters():
        param.requires_grad = value


# vae
def load_vae(
    kl_path: Path = Path("kl-f8.pt"), clip_guidance: bool = False, device: str = "cuda", use_fp16: bool = False
):
    """
    Load kl-f8 stage 1 VAE from a checkpoint.
    """
    encoder = torch.load(kl_path, map_location="cpu")
    if use_fp16:
        encoder = encoder.half()
    encoder.eval()
    encoder.to(device)
    set_requires_grad(encoder, clip_guidance)
    return encoder


# bert-text
def load_bert(bert_path: Path = Path("bert.pt"), device: str = "cuda", use_fp16: bool = False):
    """
    Load BERT from a checkpoint.
    """
    bert = BERTEmbedder(1280, 32)
    sd = torch.load(bert_path, map_location="cpu")
    bert.load_state_dict(sd)
    if use_fp16:
        bert = bert.half()
    bert.to(device)
    bert.eval()  # TODO
    set_requires_grad(bert, False)
    return bert


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


# clip
def load_clip_onnx_model(device, visual_path=None, textual_path=None):
    """
    Loads an ONNX-runtime compatible checkpoint for CLIP.
    """
    onnx_model = clip_onnx(None)
    if visual_path is not None and textual_path is not None:
        onnx_model.load_onnx(visual_path=visual_path, textual_path=textual_path, logit_scale=100.0000)
    elif visual_path is not None:
        onnx_model.load_onnx(visual_path=visual_path)
    elif textual_path is not None:
        onnx_model.load_onnx(textual_path=textual_path)
    provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
    onnx_model.start_sessions(providers=[provider])
    return onnx_model, _transform(224)


# bert context
def bert_encode_cfg(text, negative, batch_size, device, bert=None):
    """
    Returns the BERT classifier-free guidance context for a batch of text.
    """
    text_emb = bert.encode([text] * batch_size).to(device).float()
    text_blank = bert.encode([negative] * batch_size).to(device).float()
    return text_emb, text_blank


# clip context
def clip_encode_cfg_onnx(clip_model, text, negative, batch_size, device):
    """
    Returns the CLIP classifier-free guidance context for a batch of text.
    """
    text_tokens = clip.tokenize([text] * batch_size, truncate=True)
    text_tokens = text_tokens.detach().cpu().numpy().astype(np.int64)

    negative_tokens = clip.tokenize([negative] * batch_size, truncate=True)
    negative_tokens = negative_tokens.detach().cpu().numpy().astype(np.int64)

    text_emb_clip = torch.Tensor(clip_model.encode_text(text_tokens)).to(device)
    text_emb_clip_blank = torch.Tensor(clip_model.encode_text(negative_tokens)).to(
        device
    )

    text_emb_norm = text_emb_clip[0] / text_emb_clip[0].norm(dim=-1, keepdim=True)
    return text_emb_clip_blank, text_emb_clip, text_emb_norm


def prepare_edit(ldm, edit, width=256, height=256, edit_y=0, edit_x=0, device="cuda", use_fp16=True):
    """
    Given an `edit` image path, embed it and return the embedding. `edit` may be an image or a `.npy` file.
    """
    if edit.endswith('.npy'):
        with open(edit, 'rb') as f:
            im = np.load(f)
            im = torch.from_numpy(im).unsqueeze(0).to(device)

            input_image = torch.zeros(1, 4, height//8, width//8, device=device)

            y = edit_y//8
            x = edit_x//8

            ycrop = y + im.shape[2] - input_image.shape[2]
            xcrop = x + im.shape[3] - input_image.shape[3]

            ycrop = ycrop if ycrop > 0 else 0
            xcrop = xcrop if xcrop > 0 else 0

            input_image[0,:,y if y >=0 else 0:y+im.shape[2],x if x >=0 else 0:x+im.shape[3]] = im[:,:,0 if y > 0 else -y:im.shape[2]-ycrop,0 if x > 0 else -x:im.shape[3]-xcrop]
            if use_fp16:
                input_image = input_image.half()

            input_image_pil = ldm.decode(input_image)
            input_image_pil = TF.to_pil_image(input_image_pil.squeeze(0).add(1).div(2).clamp(0, 1))

            input_image *= 0.18215
    else:
        input_image_pil = Image.open(edit).convert('RGB')
        input_image_pil = ImageOps.fit(input_image_pil, (width, height))

        input_image = torch.zeros(1, 4, height//8, width//8, device=device)

        im = transforms.ToTensor()(input_image_pil).unsqueeze(0).to(device)
        im = 2*im-1
        if use_fp16:
            im = im.half()
        im = ldm.encode(im).sample()

        y = edit_y//8
        x = edit_x//8

        input_image = torch.zeros(1, 4, height//8, width//8, device=device)

        ycrop = y + im.shape[2] - input_image.shape[2]
        xcrop = x + im.shape[3] - input_image.shape[3]

        ycrop = ycrop if ycrop > 0 else 0
        xcrop = xcrop if xcrop > 0 else 0

        input_image[0,:,y if y >=0 else 0:y+im.shape[2],x if x >=0 else 0:x+im.shape[3]] = im[:,:,0 if y > 0 else -y:im.shape[2]-ycrop,0 if x > 0 else -x:im.shape[3]-xcrop]

        if use_fp16:
            input_image = input_image.half()

        input_image_pil = ldm.decode(input_image)
        input_image_pil = TF.to_pil_image(input_image_pil.squeeze(0).add(1).div(2).clamp(0, 1))

        input_image *= 0.18215
    return input_image
        


def create_cfg_fn(model, guidance_scale):
    """
    Create a classifier-free guidance function for a model with the given guidance scale.
    """

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
    """
    Logs an autoedit sample to a file.
    """
    base_dir = base_dir / prefix
    log_description = f"{prefix}_batch_{batch_index:03}_score_{score.item():.3f}"

    target_dir = base_dir.joinpath("img", f"{batch_index:3f}")
    decoded_image_path = base_dir.joinpath("img", target_dir, f"simulation_{simulation_iter}:.3f{log_description}.png")
    vae_encoding_as_npy = base_dir.joinpath("vae_npy").joinpath(log_description + ".npy")
    return decoded_image_path, vae_encoding_as_npy, score


def pack_model_kwargs(
    text_emb: torch.Tensor = None,
    text_blank: torch.Tensor = None,
    text_emb_clip: torch.Tensor = None,
    text_emb_clip_blank: torch.Tensor = None,
    image_embed: torch.Tensor = None,
    model_params: dict = None,
):
    """
    Pack model kwargs for a latent diffusion inpaint model.
    """
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


def spherical_dist_loss(
    first_vector: torch.Tensor, second_vector: torch.Tensor
) -> torch.Tensor:
    """
    Compute the spherical distance loss between two vectors.
    """
    first_vector = F.normalize(first_vector, dim=-1)
    second_vector = F.normalize(second_vector, dim=-1)
    return (first_vector - second_vector).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(batch: torch.Tensor) -> torch.Tensor:
    """L2 total variation loss, as in Mahendran et al."""
    batch = F.pad(batch, (0, 1, 0, 1), "replicate")
    x_diff = batch[..., :-1, 1:] - batch[..., :-1, :-1]
    y_diff = batch[..., 1:, :-1] - batch[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])
