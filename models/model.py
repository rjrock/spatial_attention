'''model.py'''

import torch.nn as nn


class Model(nn.Module):
    MAX    = 'max'
    SAMPLE = 'sample'
    BEAM   = 'beam'

    def __init__(self, encoder, decoder, vocab):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab

    def forward(self, image):
        return self.caption(image)

    def encode(self, image):
        return self.encoder(image)

    def embed(self, token_idx):
        return self.decoder.embed(token_idx)
