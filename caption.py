'''caption.py'''

import argparse
import json

import utils.directory as directory
import utils.image as image
import utils.storage as storage
import utils.visualization as viz

from pathlib import Path

from models import Beam


def parse_args():
    loadfile       = directory.models/'saved/03-4000.pt'
    val_annotation = directory.annotations/'val5000_img2caps.json'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image', type=str,
        help='input image for generating caption'
    )
    parser.add_argument(
        '--annotation_file', type=str, default=val_annotation,
        help='path annotation json file to sample from'
    )
    parser.add_argument('--loadfile', type=Path, default=loadfile)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model = storage.load_model(args.loadfile)
    model = model.eval()
    with open(args.annotation_file, 'r') as f:
        data = json.load(f)
    img = image.sample_random_image(data)
    beam = Beam(model)
    tokens, αs = beam.search(img, width=1, pay_attention=True)
    img = image.normalize_image(img.squeeze(axis=0))
    viz.plot_attentions(αs, tokens, img)


if __name__ == '__main__':
    main()
