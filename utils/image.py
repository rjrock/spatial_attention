'''image.py'''

import numpy as np

import utils.directory as directory

from PIL import Image
from torchvision import transforms


def load_image(path, is_transform=False):
    # Don't perform crop or horizontal flip
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))]
    )
    image = Image.open(path).convert('RGB')
    if is_transform:
        image = image.resize([224, 224], Image.LANCZOS)
        image = transform(image).unsqueeze(0)
    return image


def normalize_image(image):
    image = np.transpose(image, (1, 2, 0))
    image -= image.min()
    image /= image.max()
    return image


def sample_random_image(img2caps):
    nums = list(range(len(img2caps)))
    idx = np.random.choice(nums, size=1).squeeze(axis=0)
    imagename = list(img2caps.keys())[idx]
#   image = load_image(directory.images/'val'/imagename, is_transform=True)
    image = load_image(directory.images/'train'/imagename, is_transform=True)
    return image
