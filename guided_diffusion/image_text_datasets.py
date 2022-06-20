import io
import math
import random
from pathlib import Path

import numpy as np
import webdataset as wds
from braceexpand import braceexpand
from mpi4py import MPI
from PIL import Image


def load_data(
    *,
    data_dir,
    batch_size,
    random_crop=False,
    random_flip=True,
    image_key="jpg",
    caption_key="txt",
    cache_dir=None,
    epochs=None,
    shard_size=None,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    wds_uris = parse_data_dir(data_dir)
    print(f"Found {len(wds_uris)} tar files of total {len(wds_uris)}")

    dataset = load_webdataset(
        256, # TODO
        wds_uris,
        random_crop=random_crop,
        random_flip=random_flip,
        myimg=image_key,
        mycap=caption_key,
        cache_dir=cache_dir,
    )
    if epochs and shard_size:
        total_size = epochs * shard_size * len(wds_uris)
        print(f"Number of samples to be trained: {total_size}")
        dataset = dataset.shuffle(total_size)
    dataset = dataset.batched(batch_size)
    loader = wds.WebLoader(
        dataset, batch_size=None, shuffle=False
    )
    while True:
        yield from loader


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def clean_caption(caption):
    caption = caption.decode("utf-8")
    caption = (
        caption.replace("\n", " ")
        .replace("\t", " ")
        .replace("\r", " ")
        .replace("  ", " ")
    )
    caption = caption.strip()
    return caption


def load_webdataset(
    resolution,
    file_paths,
    random_crop=False,
    random_flip=False,
    myimg="jpg",
    mycap="txt",
    cache_dir=None,
):
    def bytes_to_pil_image(item):
        pil_image = Image.open(io.BytesIO(item)).convert("RGB")
        pil_image.load()
        return pil_image

    def filter_by_item(item):
        if mycap not in item: return False
        if myimg not in item: return False
        return True

    def pil_transform_to_np(arr):
        if random_crop:
            arr = random_crop_arr(arr, resolution, min_crop_frac=0.95) # TODO make this a param
        else:
            arr = center_crop_arr(arr, resolution)
        if random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]
        arr = arr.astype(np.float32) / 127.5 - 1
        return np.transpose(arr, [2, 0, 1])

    image_text_mapping = {myimg: bytes_to_pil_image, mycap: clean_caption}
    image_mapping = {myimg: pil_transform_to_np}
    dataset = wds.WebDataset(
        urls=file_paths,
        handler=wds.warn_and_continue,
        cache_dir=cache_dir,
        # shardshuffle=True,
        # nodesplitter=wds.split_by_worker,
    )
    filtered_dataset = dataset.select(filter_by_item)
    dataset = (
        filtered_dataset.map_dict(**image_text_mapping)
        .map_dict(**image_mapping)
        .to_tuple(myimg, mycap)
    )
    return dataset


def parse_data_dir(data_dir):
    if Path(data_dir).is_dir():
        wds_uris = [
            str(p) for p in Path(data_dir).glob("**/*") if ".tar" in str(p).lower()
        ]
        assert (
            len(wds_uris) > 0
        ), "The directory ({}) does not contain any WebDataset/.tar files.".format(
            data_dir
        )
        print(
            "Found {} WebDataset .tar(.gz) file(s) under given path {}!".format(
                len(wds_uris), data_dir
            )
        )
    elif "s3://" in data_dir.lower():
        data_dir = f"pipe:aws s3 cp {data_dir} -"
    elif ("http://" in data_dir.lower()) | ("https://" in data_dir.lower()):
        wds_uris = f"pipe:curl -L -s {data_dir} || true"
        print("Found {} http(s) link under given path!".format(len(wds_uris), data_dir))
    elif "gs://" in data_dir.lower():
        wds_uris = f"pipe:gsutil cat {data_dir} || true"
        print("Found {} GCS link under given path!".format(len(wds_uris), data_dir))

    if ".tar" in data_dir:
        wds_uris = braceexpand(data_dir)
        print("Found WebDataset .tar(.gz) file under given path {}!".format(data_dir))
    return wds_uris
