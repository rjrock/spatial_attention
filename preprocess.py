'''preprocess.py'''

import json
import pickle

from PIL import Image
from collections import defaultdict
from tqdm import tqdm

import metrics.cider
import utils.directory as directory


format_dir = directory.data/'formatted'
original_dir  = directory.data/'original'
cider_dir = directory.data/'cider'


def resize_images(indir, outdir, size):
    print(f'Resizing images in {indir} and storing to {outdir}')
    outdir.mkdir(exist_ok=True, parents=True)
    paths = list(indir.glob('*.jpg'))
    for path in tqdm(paths):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                img = img.resize(size, Image.ANTIALIAS)
                img.save(f'{outdir}/{path.name}', img.format)


def create_img2caps(data):
    id2filename = {image['id']: image['file_name'] for image in data['images']}
    img2caps = defaultdict(list)
    for ann in data['annotations']:
        image_id  = ann['image_id']
        imagename = id2filename[image_id]
        caption   = ann['caption']
        img2caps[imagename].append(caption)
    return img2caps


def create_cap2img(data):
    id2filename = {image['id']: image['file_name'] for image in data['images']}
    cap2img = []
    for ann in data['annotations']:
        image_id = ann['image_id']
        imagename = id2filename[image_id]
        caption = ann['caption']
        cap2img.append({'caption': caption, 'image': imagename})
    return cap2img


def extract_training_annotations():
    global format_dir, original_dir
    infiles = [('train', original_dir/'annotations'/'captions_train2014.json')]
    outdir = format_dir/'annotations'
    outdir.mkdir(exist_ok=True, parents=True)
    for filetype, infile in infiles:
        with open(infile, 'r') as f:
            data = json.load(f)
        cap2img = create_cap2img(data)
        cap2img_file = outdir/f'{filetype}_cap2img.json'
        with open(cap2img_file, 'w') as f:
            json.dump(cap2img, f)
        img2caps = create_img2caps(data)
        img2caps_file = outdir/f'{filetype}_img2caps.json'
        with open(img2caps_file, 'w') as f:
            json.dump(img2caps, f)


def extract_validation_annotations():
    global original_dir
    val_file = original_dir/'annotations'/'captions_val2014.json'
    data = json.load(open(val_file))
    images      = data['images']
    annotations = data['annotations']
    # Take 5000 unique images for faster validation metrics
    images = data['images'][:5000]
    image_ids = {image['id']: True for image in images}
    annotations = list(filter(lambda x: x['image_id'] in image_ids, annotations))
    data = {'images'      : images,
            'annotations' : annotations}
    cap2img = create_cap2img(data)
    cap2img_file = directory.annotations/'val5000_cap2img.json'
    json.dump(cap2img, open(cap2img_file, 'w'))
    img2caps = create_img2caps(data)
    img2caps_file = directory.annotations/'val5000_img2caps.json'
    json.dump(img2caps, open(img2caps_file, 'w'))


def resize():
    global format_dir, original_dir
    size = (256, 256)
    indirs  = [original_dir/'images'/'val2014', original_dir/'images'/'train2014']
    outdirs = [format_dir/'images'/'val'      , format_dir/'images'/'train']
    for indir, outdir in zip(indirs, outdirs):
        resize_images(indir, outdir, size)


def create_cider_ngrams():
    val5000_img2caps = format_dir/'annotations'/'val5000_img2caps.json'
    val5000_outfile = cider_dir/'val5000_img2caps_ngram.pkl'
    cider_dir.mkdir(exist_ok=True)
    with open(val5000_img2caps, 'r') as f:
        data = json.load(f)
    corpus = metrics.cider.extract_document_frequency(data)
    print(f'Extracting cider n-grams from {val5000_outfile}')
    with open(val5000_outfile, 'wb') as f:
        pickle.dump(corpus, f)
        print(f'Wrote cider data to {val5000_outfile}')


def main():
    resize()
    extract_training_annotations()
    extract_validation_annotations()
    create_cider_ngrams()


if __name__ == '__main__':
    main()
