'''train.py'''

import argparse
import numpy as np
import torch
import torch.nn.functional as F

import utils.data_loader as data_loader
import utils.directory as directory
import utils.settings as settings
import utils.storage as storage

from pathlib import Path
from torch.nn.utils.rnn import pack_padded_sequence


def parse_args():
    loadfile = directory.models/'saved/06-9999.pt'
#   loadfile = Path('none')
    save_dir = directory.models/'saved'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--embedding_path', type=str,
        default=f'{directory.embedding}/glove256_vocab.kv',
        help='Word embedding weights'
    )
    parser.add_argument(
        '--log_step', type=int, default=10,
        help='step size for prining log info'
    )
    parser.add_argument(
        '--save_step', type=int, default=1000,
        help='step size for saving trained models'
    )
    parser.add_argument('--batch_size'   , type=int  , default=64)
    parser.add_argument('--fine_tune'    , type=bool , default=False)
    parser.add_argument('--is_xe_loss'   , type=bool , default=True)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--loadfile'     , type=Path , default=loadfile)
    parser.add_argument('--num_epochs'   , type=int  , default=100)
    parser.add_argument('--num_workers'  , type=int  , default=8)
    parser.add_argument('--save_dir'     , type=Path , default=save_dir)
    args = parser.parse_args()
    args.formatted = directory.formatted
    args.save_dir.mkdir(exist_ok=True, parents=True)
    return args


def calculate_xe_loss(encoder, decoder, images, padded_inputs, padded_targets, lengths):
    features = encoder(images)
    padded_logits = decoder(features, padded_inputs, lengths)
    packed_logits = pack_padded_sequence(padded_logits, lengths, batch_first=True)
    packed_targets = pack_padded_sequence(padded_targets, lengths, batch_first=True)
    xe_loss = F.cross_entropy(packed_logits.data, packed_targets.data)
    return xe_loss


def train(encoder, decoder, data, args, vocab):
    encoder = encoder.train()
    decoder = decoder.train()
    params = (list(decoder.parameters())
              + list(encoder.project_global.parameters())
              + list(encoder.project_spatial.parameters()))
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    total_step = len(data)
    total_epochs = args.base_epoch + args.num_epochs
    for i in range(args.num_epochs):
        epoch = args.base_epoch + i
        for step, (images, padded_inputs, padded_targets,
                   lengths, refs) in enumerate(data):
            images = images.to(settings.device)
            padded_inputs = padded_inputs.to(settings.device)
            padded_targets = padded_targets.to(settings.device)
            loss = 0
            if args.is_xe_loss:
                xe_loss = calculate_xe_loss(encoder, decoder, images, padded_inputs,
                                            padded_targets, lengths)
                loss += xe_loss
                perplexity = np.exp(xe_loss.item())
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()
            if (step+1) % args.log_step == 0:
                update = (f'Epoch [{epoch}/{total_epochs}],'
                          f' Step [{step+1}/{total_step}],')
                if args.is_xe_loss:
                    perplexity = np.exp(xe_loss.item())
                    update += (f' XE Loss [{xe_loss.item():.4f}],'
                               f' Perplexity [{perplexity:5.4f}]')
                print(update)
            if (step+1) % args.save_step == 0:
                storage.save_checkpoint(args.save_dir, epoch, 0, encoder, decoder)
        storage.save_checkpoint(args.save_dir, epoch, 9999, encoder, decoder)


def main():
    args = parse_args()
    encoder, decoder = storage.load_models(args.loadfile)
    if args.fine_tune:
        print('Setting layers 3 and 4 of ResNet trainable')
        encoder.resnet.set_tunable()
    vocab = storage.load_vocab()
    args.base_epoch = storage.load_base_epoch(args.loadfile)
    train_image_dir = directory.images/'train'
    train_cap2img = directory.annotations/'train_cap2img.json'
    train_img2caps = directory.annotations/'train_img2caps.json'
    train_data = data_loader.get_loader(
        train_image_dir, train_cap2img.as_posix(),
        train_img2caps.as_posix(), vocab, args.batch_size,
        shuffle=True, num_workers=args.num_workers
    )
    train(encoder, decoder, train_data, args, vocab)


if __name__ == '__main__':
    main()
