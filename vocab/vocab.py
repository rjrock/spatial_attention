'''vocab.py'''


class Vocabulary:
    pad   = '<pad>'
    start = '<start>'
    end   = '<end>'
    unk   = '<unk>'

    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}
        self.idx = 0
        metatokens = [self.pad, self.start, self.end, self.unk]
        self.add_tokens(metatokens)
        self.start_idx = self.token2idx[self.start]
        self.end_idx = self.token2idx[self.end]

    def add_token(self, token):
        if token not in self.token2idx:
            self.token2idx[token] = self.idx
            self.idx2token[self.idx] = token
            self.idx += 1

    def add_tokens(self, tokens):
        [self.add_token(token) for token in tokens]

    def idxs2tokens(self, idxs):
        return [self.idx2token[idx] for idx in idxs]

    def __call__(self, token):
        if token not in self.token2idx:
            return self.token2idx[self.unk]
        return self.token2idx[token]

    def __len__(self):
        return self.idx
