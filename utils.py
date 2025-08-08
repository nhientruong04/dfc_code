# General imports used throughout the tutorial
# file operations
import os
from typing import List, Optional
from loguru import logger

import numpy as np
from tqdm import tqdm
from PIL import Image

# from hailo_sdk_client import ClientRunner, InferenceContext


def build_calib_dataset(image_folders: List[str], num: int = 1024,
                        output_height: int = 256, output_width: int = 256) -> np.array:
    norm = False
    calib_dataset = list()
    total_images = 0

    for image_folder in image_folders:
        print(f'Get images from {image_folder}')
        dataset = get_dataset(image_folder, norm, output_height, output_width)
        total_images += dataset.shape[0]
        calib_dataset.append(dataset)

    if total_images < num:
        logger.warning(f'Total of images {total_images} received is less than '
                       'the expected number of items for calib.')
    else:
        logger.info(f'Build calib dataset successfully total {total_images} '
                    'the dataset will be clipped to {num}.')

    return \
        np.concatenate(calib_dataset) if total_images < num \
        else np.concatenate(calib_dataset)[:1024]


def get_dataset(image_folder: str, norm: bool,
                output_height: int = 256, output_width: int = 256) -> np.array:
    assert os.path.exists(image_folder), "Path not exist"
    image_list = os.listdir(image_folder)
    image_paths = [os.path.join(image_folder, i) for i in image_list]

    images = np.empty((len(image_list), output_height,
                      output_width, 3), dtype=np.uint8)

    description = 'Building normed dataset' if norm else 'Building un-normed dataset'

    for i, path in enumerate(tqdm(image_paths, desc=description)):
        try:
            image = Image.open(path)
            image = image.resize(
                (output_width, output_height), Image.Resampling.BILINEAR)
            if norm:
                pass

            image = np.array(image)
            images[i, :, :, :] = image
        except Exception as e:
            print(f"An error occured {e}")
            break

    return images


if __name__ == "__main__":
    # image_folder = '/home/nhien/mvtec-ad/bottle/train/good/'
    # get_dataset(image_folder, norm=False,
    #             output_height=256, output_width=256)
    image_folders = ['/home/nhien/mvtec-ad/bottle/train/good/',
                     '/home/nhien/mvtec-ad/cable/train/good/']

    calib_databset = build_calib_dataset(image_folders)
    print(calib_databset.shape)
