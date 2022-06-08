# Finetune Latent Diffusion

This repo is modified from [glid-3-xl](https://github.com/jack000/glid-3-xl).

<a href="https://replicate.com/laion-ai/ongo" target="_blank"><img src="https://img.shields.io/static/v1?label=run&message=ongo&color=blue"></a> <a href="https://replicate.com/laion-ai/erlich" target="_blank"><img src="https://img.shields.io/static/v1?label=run&message=erlich&color=orange"></a> 


Checkpoints are finetuned from `glid-3-xl` [inpaint.pt](https://dall-3.com/models/glid-3-xl/inpaint.pt)

Aesthetic CLIP embeds are provided by [aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor)

## Install

### virtual environment:

```bash
python3 -m venv .venv
source venv/bin/activate
(venv) $
```

### pytorch

```bash
(venv) $ pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

### latent-diffusion/taming-transformers
```bash

(venv) $ git clone https://github.com/CompVis/latent-diffusion.git
(venv) $ git clone https://github.com/CompVis/taming-transformers
(venv) $ pip install -e ./taming-transformers
(venv) $ pip install omegaconf>=2.0.0 pytorch-lightning>=1.0.8 torch-fidelity einops

```

### ldm-finetune
    
```bash
(venv) $ git clone https://github.com/laion-ai/ldm-finetune.git
(venv) $ cd ldm-finetune
(venv) $ pip install -e .
```

## Checkpoints 

```
# text encoder (required)
(venv) $ wget https://dall-3.com/models/glid-3-xl/bert.pt

# ldm first stage (required)
(venv) $ wget https://dall-3.com/models/glid-3-xl/kl-f8.pt

# there are several diffusion models to choose from:

# original diffusion model from CompVis
(venv) $ wget https://dall-3.com/models/glid-3-xl/diffusion.pt

# new model fine tuned on a cleaner dataset (will not generate watermarks, split images or blurry images)
(venv) $ wget https://dall-3.com/models/glid-3-xl/finetune.pt

# inpaint
(venv) $ wget https://dall-3.com/models/glid-3-xl/inpaint.pt

# erlich
(venv) $ wget -O erlich.pt https://huggingface.co/laion/erlich/raw/main/model/ema_0.9999_120000.pt

# ongo
(venv) $ wget https://huggingface.co/laion/ongo/resolve/main/ongo.pt

```

## Generating images

```
# fast PLMS sampling
(venv) $ python sample.py --model_path finetune.pt --batch_size 6 --num_batches 6 --text "a cyberpunk girl with a scifi neuralink device on her head"

# classifier free guidance + CLIP guidance (better adherence to prompt, much slower)
(venv) $ python sample.py --clip_guidance --model_path finetune.pt --batch_size 1 --num_batches 12 --text "a cyberpunk girl with a scifi neuralink device on her head | trending on artstation"

# sample with an init image
(venv) $ python sample.py --init_image picture.jpg --skip_timesteps 10 --model_path finetune.pt --batch_size 6 --num_batches 6 --text "a cyberpunk girl with a scifi neuralink device on her head"

# generated images saved to ./output/
# generated image embeddings saved to ./output_npy/ as npy files
```


## Editing images
aka human guided diffusion. You can use inpainting to generate more complex prompts by progressively editing the image

note: you can use > 256px but the model only sees 256x256 at a time, so ensure the inpaint area is smaller than that

note: inpaint training wip
```

# install PyQt5 if you want to use a gui, otherwise supply a mask file
pip install PyQt5

# this will pop up a window, use your mouse to paint
# use the generated npy files instead of png for best quality
(venv) $ python sample.py --model_path inpaint.pt --edit output_npy/00000.npy --batch_size 6 --num_batches 6 --text "your prompt"

# after painting, the mask is saved for re-use
(venv) $ python sample.py --mask mask.png --model_path inpaint.pt --edit output_npy/00000.npy --batch_size 6 --num_batches 6 --text "your prompt"

# additional arguments for uncropping
(venv) $ python sample.py --edit_x 64 --edit_y 64 --edit_width 128 --edit_height 128 --model_path inpaint.pt --edit output_npy/00000.npy --batch_size 6 --num_batches 6 --text "your prompt"

# autoedit uses the inpaint model to give the ldm an image prompting function (that works differently from --init_image)
# it continuously edits random parts of the image to maximize clip score for the text prompt
(venv) $ python autoedit.py --edit image.png --model_path inpaint.pt --batch_size 6 --text "your prompt"

```

## Training/Fine tuning

```
# batch size > 1 required
MODEL_FLAGS="--dropout 0.1 --ema_rate 0.9999 --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 32 --learn_sigma False --noise_schedule linear --num_channels 320 --num_heads 8 --num_res_blocks 2 --resblock_updown False --use_fp16 True --use_scale_shift_norm False"
TRAIN_FLAGS="--lr --batch_size 64 --microbatch 1 --log_interval 1 --save_interval 5000 --kl_model kl-f8.pt --bert_model bert.pt --resume_checkpoint diffusion.pt"
export OPENAI_LOGDIR=./logs/
export TOKENIZERS_PARALLELISM=false
python scripts/image_train_inpaint.py --data_dir /path/to/data $MODEL_FLAGS $TRAIN_FLAGS
```
