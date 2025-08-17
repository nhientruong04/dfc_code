# General imports used throughout the tutorial
# file operations
import os
from loguru import logger

import numpy as np
from typing import List
from tqdm import tqdm
from PIL import Image


def build_dataset(image_folders: List[str], calib: bool = False,
                  output_height: int = 256, output_width: int = 256) -> np.array:
    norm: bool = False
    dataset: List = list()
    total_images: int = 0

    for image_folder in image_folders:
        print(f'Get images from {image_folder}')
        data = get_dataset(image_folder, norm, output_height, output_width)
        total_images += data.shape[0]
        dataset.append(data)

    if calib:

        if total_images < 1024:
            logger.warning(f'Total of images {total_images} received is less than '
                           'the expected number of items for calib.')
            return np.concatenate(dataset)
        else:
            logger.info(f'Build calib dataset successfully with total {total_images} '
                        f'the dataset will be clipped to {1024}.')
            return np.concatenate(dataset)[:1024]

    else:

        logger.info(f'Build dataset successfully with total {total_images}.')
        return np.concatenate(dataset)


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
    image_folders = ['/home/nhien/mvtec-ad/bottle/train/good/',
                     '/home/nhien/mvtec-ad/cable/train/good/',
                     '/home/nhien/mvtec-ad/wood/train/good/']

    calib_databset = build_dataset(image_folders, calib=True)
    print(calib_databset.shape)

    np.save('test_bottle_dataset.npy', calib_databset)
