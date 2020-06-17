'''decoder.py'''

import torch
import torch.nn as nn

import utils.settings as settings

from models.attention import SpatialAttention


class Decoder(nn.Module):
    def __init__(self, vocab, k, d, embed_size=256):
        '''embed_size should be kept at 256 if using GloVe embeddings.'''
        super(Decoder, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding = nn.Embedding(self.vocab_size, embed_size)
        self.f_init_h = nn.Linear(d, d)
        self.f_init_m = nn.Linear(d, d)
        self.lstm_cell = nn.LSTMCell(embed_size + d, d)
        self.spatial_attention = SpatialAttention(k, d)
        self.W_p = nn.Linear(d, self.vocab_size)

    def forward(self, features, captions, lengths):
        h, m = self.initialize_state(features.global_)
        batch_size = features.spatial.size(0)
        embeddings = self.embed(captions)
        logits = torch.zeros(batch_size, max(lengths),
                             self.vocab_size).to(settings.device)
        # hₚ means hₜ₋₁
        for t in range(max(lengths)):
            batch_size = sum([l > t for l in lengths])
            hₚ = h[:batch_size]
            mₚ = m[:batch_size]
            w = embeddings[:batch_size, t, :]
            V = features.spatial[:batch_size]
            vᵍ = features.global_[:batch_size]
            x = torch.cat([w, vᵍ], dim=1)
            h, m = self.lstm_cell(x, (hₚ, mₚ))
            c, _ = self.spatial_attention(V, h)
            logits[:batch_size, t, :] = self.W_p(c + h)
        return logits

    def embed(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if not x.is_cuda:
            x = x.to(settings.device)
        return self.embedding(x)

    def set_embedding_weights(self, glove):
        for idx, token in enumerate(self.vocab.token2idx):
            weight = glove[token]
            self.embed.weight[idx].data = torch.from_numpy(weight)

    def initialize_state(self, global_features):
        h = self.f_init_h(global_features)
        m = self.f_init_m(global_features)
        return h, m
