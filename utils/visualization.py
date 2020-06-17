'''visualization.py'''

import PIL
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from pathlib import Path


def plot_attention(α, token, img, i):
    rows = 2
    cols = 10
    α = α.reshape((7, 7))
    weights = Image.fromarray(α)
    weights = weights.resize((224, 224), resample=PIL.Image.BILINEAR)
    plt.subplot(rows, cols, i)
    plt.imshow(img)
    if i != 1:
        plt.imshow(np.asarray(weights), alpha=0.7)
        plt.set_cmap(cm.Greys_r)
    plt.xlabel(f'{token}')
    plt.xticks([])
    plt.yticks([])


def plot_attentions(αs, tokens, img):
    plt.figure(figsize=(15, 5))
    for i, (α, token) in enumerate(zip(αs, tokens)):
        plot_attention(α, token, img, i+1)
    plt.subplots_adjust(wspace=0, hspace=-0.3)
    idx = np.random.randint(1000)
    attention_maps = Path('attention_maps')
    attention_maps.mkdir(exist_ok=True)
    filepath = f'{attention_maps}/{idx}.jpg'
    plt.savefig(filepath)
    print(f'Saved attention map to {filepath}')
