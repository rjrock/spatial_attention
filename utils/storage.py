'''storage.py'''

import pickle
import torch

from utils import directory, settings

from gensim.models.keyedvectors import KeyedVectors

from models import Decoder, Encoder, Model


def load_models(loadfile):
    vocab = load_vocab()
    encoder = Encoder()
    decoder = Decoder(vocab, k=49, d=512)
    glovefile = directory.embedding/'glove_vocab.kv'
    if loadfile.exists():
        state = torch.load(loadfile)
        decoder = state['decoder']
        encoder = state['encoder']
        print(f'Loaded from {loadfile}')
    elif glovefile.exists():
        glove = load_glove(glovefile)
        decoder.set_embedding_weights(glove)
        print('Did not load saved model. Setting word embedding weights')
    encoder = encoder.to(settings.device)
    decoder = decoder.to(settings.device)
    return encoder, decoder


def load_model(loadfile):
    encoder, decoder = load_models(loadfile)
    vocab = load_vocab()
    model = Model(encoder, decoder, vocab)
    model = model.to(settings.device)
    return model


def load_base_epoch(loadfile):
    base_epoch = 0
    if loadfile.exists():
        state = torch.load(loadfile)
        base_epoch = state['epoch'] + 1
    return base_epoch


def load_vocab(path=f'{directory.vocab}/vocab.pkl'):
    return pickle.load(open(path, 'rb'))


def load_cider_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['df'], data['num_images']


def load_glove(path):
    glove = KeyedVectors.load(path.as_posix(), mmap='r')
    return glove


def save_checkpoint(dir_, epoch, steps, encoder, decoder, verbose=True):
    state = {'epoch'  : epoch,
             'encoder': encoder,
             'decoder': decoder}
    filename = dir_/f'{epoch:02}-{steps:04}.pt'
    torch.save(state, filename)
    if verbose:
        print(f'Saved to {filename}')
