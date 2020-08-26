'''model.py'''

import numpy as np
import torch
import torch.nn as nn

import utils.settings as settings


class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, image):
        return self.max_caption(image)

    def max_caption(self, images):
        training = self.encoder.training
        self.encoder.eval()
        self.decoder.eval()
        batch_size = images.shape[0]
        features = self.encoder(images)
        h, m = self.decoder.initialize_state(features.global_)
        tokens = np.zeros((batch_size, 20), dtype=np.intc)
        token = torch.tensor([self.decoder.vocab.start_idx])
        token = token.repeat(batch_size).to(settings.device)
        for i in range(20):
            w = self.decoder.embed(token)
            x = torch.cat([w, features.global_], dim=1)
            h, m = self.decoder.lstm_cell(x, (h, m))
            c, _ = self.decoder.spatial_attention(features.spatial, h)
            logits = self.decoder.W_p(c + h)
            _, token = torch.max(logits, dim=1)
            tokens[:, i] = token.detach().cpu().numpy()
        captions = self.decode(tokens)
        if training:
            self.encoder.train()
            self.decoder.train()
        return captions

    def decode(self, array_of_tokens):
        deciphered = []
        for tokens in array_of_tokens:
            end = np.where(tokens == self.decoder.vocab.end_idx)
            try:
                end = end[0][0] + 1
            except IndexError:
                end = 20
            tokens = self.decoder.vocab.idxs2tokens(tokens[:end])
            deciphered.append(tokens)
        return deciphered

    def encode(self, image):
        return self.encoder(image)

    def embed(self, token_idx):
        return self.decoder.embed(token_idx)
