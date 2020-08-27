'''beam.py'''

import numpy as np
import torch
import torch.nn.functional as F


class Beam:
    def __init__(self, model):
        self.decoder = model.decoder
        self.encoder = model.encoder
        self.end     = model.decoder.vocab.end_idx
        self.start   = model.decoder.vocab.start_idx
        self.vocab   = model.decoder.vocab

    def search(self, image, width=3, pay_attention=False):
        features = self.encoder(image)
        tokens = [self.start]
        paths = [BeamPath(self.decoder, tokens, features)]
        seen_end = False
        j = 0
        idx = 0
        while j < 20 and not seen_end:
            paths = self.step(paths, width)
            if self.end in [path.leading_token for path in paths]:
                seen_end = True
                idx = [path.leading_token for path in paths].index(self.end)
            j += 1
        tokens = self.vocab.idxs2tokens(paths[idx].tokens)
        if pay_attention:
            return tokens, paths[idx].αs
        return tokens

    def step(self, paths, width):
        next_paths = []
        for path in paths:
            next_paths.extend(path.step(width))
        # Since the least likely paths have the most negative value (as
        # a consequence of taking the log of the probability), the
        # sorting should be reversed if we take the most likely $width paths
        next_paths = sorted(next_paths, reverse=True)
        return next_paths[:width]


class BeamPath:
    def __init__(self, decoder=None, tokens=None, features=None, logprob=0):
        if not decoder:
            # For overloading __init__
            return
        self.decoder = decoder
        self.features = features
        self.h_m = self.decoder.initialize_state(features.global_)
        self.logprob = logprob
        self.tokens = tokens
        self.αs = [np.zeros(49)]

    @property
    def h(self):
        return self.h_m[0]

    @property
    def leading_token(self):
        return self.tokens[-1]

    def __lt__(self, o):
        return self.logprob < o.logprob

    def copy(self):
        o = BeamPath()
        o.decoder = self.decoder
        o.h_m = self.h_m
        o.tokens = [token for token in self.tokens]
        o.features = self.features
        o.logprob = self.logprob
        o.αs = [α for α in self.αs]
        return o

    def step(self, width):
        embedding = self.decoder.embed([self.leading_token])
        x = torch.cat([embedding, self.features.global_], dim=1)
        h, m = self.decoder.lstm_cell(x, self.h_m)
        self.h_m = (h, m)
        leaves = self.branch(width)
        return leaves

    def branch(self, width):
        probs, tokens, α = self.top_k(k=width)
        leaves = []
        for token, prob in zip(tokens, probs):
            leaf = self.copy()
            leaf.αs.append(α)
            leaf.tokens.append(token)
            leaf.logprob += np.log(prob)
            leaves.append(leaf)
        return leaves

    def top_k(self, k):
        c, α = self.decoder.spatial_attention(self.features.spatial, self.h)
        logits = self.decoder.W_p(c + self.h)
        probs = F.softmax(logits, dim=1)
        probs, tokens = torch.topk(probs, k, dim=1)
        # Squeeze, since beam search runs on a single image at a time
        probs  = probs.squeeze(0).detach().cpu().numpy()
        tokens = tokens.squeeze(0).detach().cpu().numpy()
        α      = α.squeeze(0).detach().cpu().numpy()
        return probs, tokens, α
