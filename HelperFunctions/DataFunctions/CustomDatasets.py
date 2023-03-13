import os
import pathlib
import random

import numpy as np
import torch.optim.lr_scheduler
from PIL import Image
from torch.utils.data import Dataset



class ImageDataset(Dataset):
    def __init__(self, root: str, folder: str, klass: int, extension: str = "jpg", transform=None):
        self._data = pathlib.Path(root) / folder
        self.klass = klass
        self.extension = extension
        self.transform = transform
        # Only calculate once how many files are in this folder
        # Could be passed as argument if you precalculate it somehow
        # e.g. ls | wc -l on Linux
        self._length = sum(1 for entry in os.listdir(self._data))

    def __len__(self):
        # No need to recalculate this value every time
        return self._length

    def __getitem__(self, index):
        # images always follow [0, n-1], so you access them directly
        image = Image.open(self._data / "{}.{}".format(str(index), self.extension)).convert('RGB')
        if self.transform:
            image = self.transform(image)

        sample = image, torch.tensor(self.klass)
        return sample


class GetJigsawPuzzleDataset(Dataset):
    # Dataset used for training the Jigsaw Puzzle Pretext Task

    def __init__(self, root: str, folder: str, avail_permuts_file_path, range_permut_indices=None, transform=None,
                 extension: str = "jpg"):
        self._data = pathlib.Path(root) / folder
        self.extension = extension
        self.transform = transform
        self.permuts_avail = np.load(avail_permuts_file_path)
        self.range_permut_indices = range_permut_indices
        self._length = sum(1 for entry in os.listdir(self._data))

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        # 1. Image choosen --> Random Crop
        # 2. Crop 9 Patches --> Generate permutated patch Array
        # 3. Apply Data Transforms to 9 Patches --> Return as Tensor

        # Select sample
        pil_image = Image.open(self._data / "{}.{}".format(str(index), self.extension)).convert('RGB')

        # Convert image to torch tensor
        pil_image = pil_image.resize((256, 256))
        pil_image = randomCrop(pil_image, 225)

        # Get nine crops for the image
        nine_crops = get_nine_crops(pil_image)

        # Permut the 9 patches obtained from the image
        if self.range_permut_indices:
            permut_ind = random.randint(self.range_permut_indices[0], self.range_permut_indices[1])
        else:
            permut_ind = random.randint(0, len(self.permuts_avail) - 1)

        permutation_config = self.permuts_avail[permut_ind]

        permuted_patches_arr = [None] * 9
        for crop_new_pos, crop in zip(permutation_config, nine_crops):
            permuted_patches_arr[crop_new_pos] = crop

        # Apply data transforms
        tensor_patches = torch.zeros(9, 3, 64, 64)
        for ind, jigsaw_patch in enumerate(permuted_patches_arr):
            jigsaw_patch_tr = self.transform(jigsaw_patch)
            tensor_patches[ind] = jigsaw_patch_tr

        return tensor_patches, permut_ind

def get_nine_crops(pil_image):
    """Get nine crops for a square pillow image. That is height and width of the image should be same.
    :param pil_image: pillow image
    :return: List of pillow images. The nine crops
    """
    w, h = pil_image.size
    diff = int(w / 3)

    r_vals = [0, diff, 2 * diff]
    c_vals = [0, diff, 2 * diff]

    list_patches = []

    for r in r_vals:
        for c in c_vals:
            left = c
            top = r
            right = c + diff
            bottom = r + diff

            patch = pil_image.crop((left, top, right, bottom))
            list_patches.append(patch)

    return list_patches


def randomCrop(image, size):
    """random crops given image

    :param image: pil_image used for random crop
    :param size: size for the random crop
    :return: returns random cropped pil_image
    """
    x, y = image.size

    left = random.randint(0, x - size)
    top = random.randint(0, y - size)
    right = left + size
    bottom = top + size

    return image.crop((left, top, right, bottom))
