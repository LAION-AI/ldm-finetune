# Finetune Latent Diffusion

This repo is modified from [glid-3-xl](https://github.com/jack000/glid-3-xl).  Aesthetic CLIP embeds are provided by [aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor)

<a href="https://replicate.com/laion-ai/ongo" target="_blank"><img src="https://img.shields.io/static/v1?label=run&message=ongo&color=blue"></a>

<img src="/assets/ongo-painting-of-a-farm-with-flowers.png" width="256"></img>

<a href="https://replicate.com/laion-ai/erlich" target="_blank"><img src="https://img.shields.io/static/v1?label=run&message=erlich&color=orange"></a>

<img src="/assets/colorful-glowing-low-poly-logo-of-a-lion.png" width="256"></img>


## Prerequisites

Please ensure the following dependencies are installed prior to building this repo:

- software-properties-common
- build-essential
- libopenmpi-dev
- liblzma-dev
- libnss3-dev
- zlib1g-dev
- libgdbm-dev
- libncurses5-dev
- libssl-dev
- libffi-dev
- libbz2-dev

### Pytorch

It's a good idea to use a virtual environment or a conda environment.

```bash
python3 -m venv .venv
source venv/bin/activate
(venv) $
```

Before installing, you should install pytorch manually by following the instructions at [pytorch.org](https://pytorch.org/get-started/locally/)

In my instance, I needed the following for cuda 11.3.

```bash
(venv) $ pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```

To check your cuda version, run `nvidia-smi`.

### Install ldm-finetune

You can now install this repo by running `pip install -e .` in the project directory.

```bash
(venv) $ git clone https://github.com/laion-ai/ldm-finetune.git
(venv) $ cd ldm-finetune
(venv) $ pip install -e .
```

## Checkpoints

## CLIP ViT-L/14 - ONNX
```sh
wget -O textual.onnx 'https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/ViT-L/14/textual.onnx'
wget -O visual.onnx 'https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/ViT-L-14/visual.onnx'
```

### BERT Text Encoder

```sh
wget https://dall-3.com/models/glid-3-xl/bert.pt
```

### Latent Diffusion Stage 1 (vqgan)

```sh
wget https://dall-3.com/models/glid-3-xl/kl-f8.pt
```

### Latent Diffusion Stage 2 (diffusion)

There are several stage 2 checkpoints to choose from:

#### CompVis - `diffusion.pt`

The original checkpoint from CompVis trained on `LAION-400M`.

```sh
wget https://dall-3.com/models/glid-3-xl/diffusion.pt
```

#### jack000 - `finetune.pt`

The first finetune from jack000's [glid-3-xl](https://github.com/jack000/glid-3-xl). Modified to accept a CLIP text embed and finetuned on curated data to help with watermarks. Doesn't support inpainting.

```sh
wget https://dall-3.com/models/glid-3-xl/finetune.pt 
```

#### jack000 - `inpaint.pt`

This second finetune adds support for inpainting and can be used for unconditional output as well by setting the inpaint `image_embed` to zeros.

wget https://dall-3.com/models/glid-3-xl/inpaint.pt

#### LAION - `erlich.pt`

`erlich` is [inpaint.pt](https://dall-3.com/models/glid-3-xl/inpaint.pt) finetuned on a dataset collected from LAION-5B named `Large Logo Dataset`. It consists of roughly 100K images of logos with captions generated via BLIP using aggressive re-ranking and filtering.

```sh
wget -O erlich.pt https://huggingface.co/laion/erlich/resolve/main/model/ema_0.9999_120000.pt
```

> ["You know aviato?"](https://www.youtube.com/watch?v=7Q9nQXdzNd0&t=39s)

#### LAION - `ongo.pt`

ONGO is [inpaint.pt](https://dall-3.com/models/glid-3-xl/inpaint.pt) finetuned on the Wikiart dataset consisting of about 100K paintings with captions generated via BLIP using aggressive re-ranking and filtering. We also make use of the original captions which contain the author name and the painting title. 

```sh
wget https://huggingface.co/laion/ongo/resolve/main/ongo.pt
```

> ["Ongo Gablogian, the art collector. Charmed, I'm sure."](https://www.youtube.com/watch?v=CuMO5q1Syek)

## Generating images

```bash
# fast PLMS sampling
(venv) $ python sample.py --model_path erlich.pt --batch_size 6 --num_batches 6 --text "a cyberpunk girl with a scifi neuralink device on her head"

# classifier free guidance + CLIP guidance (better adherence to prompt, much slower)
(venv) $ python sample.py --clip_guidance --model_path finetune.pt --batch_size 1 --num_batches 12 --text "a cyberpunk girl with a scifi neuralink device on her head | trending on artstation"

# sample with an init image
(venv) $ python sample.py --init_image picture.jpg --skip_timesteps 10 --model_path ongo.pt --batch_size 6 --num_batches 6 --text "a cyberpunk girl with a scifi neuralink device on her head"
```

## Editing images

aka human guided diffusion. You can use inpainting to generate more complex prompts by progressively editing the image

note: you can use > 256px but the model only sees 256x256 at a time, so ensure the inpaint area is smaller than that

```bash
# install PyQt5 if you want to use a gui, otherwise supply a mask file
(venv) $ pip install PyQt5

# this will pop up a window, use your mouse to paint
# use the generated npy files instead of png for best quality
(venv) $ python sample.py --model_path inpaint.pt --edit output_npy/00000.npy --batch_size 6 --num_batches 6 --text "your prompt"

# after painting, the mask is saved for re-use
(venv) $ python sample.py --mask mask.png --model_path inpaint.pt --edit output_npy/00000.npy --batch_size 6 --num_batches 6 --text "your prompt"

# additional arguments for uncropping
(venv) $ python sample.py --edit_x 64 --edit_y 64 --edit_width 128 --edit_height 128 --model_path inpaint.pt --edit output_npy/00000.npy --batch_size 6 --num_batches 6 --text "your prompt"
```

## Autoedit 

> Autoedit uses the inpaint model to give the ldm an image prompting function (that works differently from --init_image)
> It continuously edits random parts of the image to maximize clip score for the text prompt

```bash
CUDA_VISIBLE_DEVICES=5 python autoedit.py \
    --model_path erlich_on_pokemon_logs_run2/model017000.pt  --kl_path kl-f8.pt --bert_path bert.pt \
    --text "high quality professional pixel art" --negative "" --prefix autoedit_debug \
    --batch_size 64 --width 256 --height 256 --iterations 25 \
    --starting_threshold 0.6 --ending_threshold 0.5 \
    --starting_radius 5 --ending_radius 0.1 \
    --seed -1 --guidance_scale 5.0 --steps 30 \
    --aesthetic_rating 9 --aesthetic_weight 0.5 --wandb_name autoedit_pixelart
```


## Training/Fine tuning

See the script below for an example of finetuning your own model from one of the available chekcpoints. 


Finetuning Tips/Tricks
* NVIDIA GPU required. You will need an A100 or better to use a batch size of 64. Using less may present stability issues.
* Monitor the `grad_norm` in the output log.  If it ever goes above 1.0 the checkpoint may be ruined due to exploding gradients. 
    * to fix, try reducing the learning rate, decreasing the batch size.
    * Train in 32-bit
    * Resume with saved optimizer state when possible.

```bash
#!/bin/bash
# Finetune glid-3-xl inpaint.pt on your own webdataset.
# Note: like all one-off scripts, this is likely to become out of date at some point.
# running python scripts/image_train_inpaint.py --help will give you more info.

# model flags
use_fp16=False # TODO can cause more trouble than it's worth.
MODEL_FLAGS="--dropout 0.1 --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 32 --learn_sigma False --noise_schedule linear --num_channels 320 --num_heads 8 --num_res_blocks 2 --resblock_updown False --use_fp16 $use_fp16 --use_scale_shift_norm False"

# checkpoint flags
resume_checkpoint="inpaint.pt"
kl_model="kl-f8.pt"
bert_model="bert.pt"

# training flags
epochs=80
shard_size=512
batch_size=32
microbatch=-1
lr=1e-6 # lr=1e-5 seems to be stable. going above 3e-5 is not stable.
ema_rate=0.9999 # TODO you may want to lower this to 0.999, 0.99, 0.95, etc.
random_crop=False
random_flip=False
cache_dir="cache"
image_key="jpg"
caption_key="txt"
data_dir=/my/custom/webdataset/ # TODO set this to a real path

# interval flags
sample_interval=100
log_interval=1
save_interval=2000

CKPT_FLAGS="--kl_model $kl_model --bert_model $bert_model --resume_checkpoint $resume_checkpoint"
INTERVAL_FLAGS="--sample_interval $sample_interval --log_interval $log_interval --save_interval $save_interval"
TRAIN_FLAGS="--epochs $epochs --shard_size $shard_size --batch_size $batch_size --microbatch $microbatch --lr $lr --random_crop $random_crop --random_flip $random_flip --cache_dir $cache_dir --image_key $image_key --caption_key $caption_key --data_dir $data_dir"
COMBINED_FLAGS="$MODEL_FLAGS $CKPT_FLAGS $TRAIN_FLAGS $INTERVAL_FLAGS"
export OPENAI_LOGDIR=./erlich_on_pixel_logs_run6_part2/
export TOKENIZERS_PARALLELISM=false

# TODO comment out a line below to train either on a single GPU or multi-GPU
# single GPU
# python scripts/image_train_inpaint.py $COMBINED_FLAGS

# or multi-GPU
# mpirun -n 8 python scripts/image_train_inpaint.py $COMBINED_FLAGS
```