'''bleu.py'''

import nltk

from metrics.metric import Metric


class Bleu(Metric):
    def __init__(self, m=4):
        self.m = m

    def score(self, refs, generated):
        smoothing = nltk.translate.bleu_score.SmoothingFunction()
        weights = [1/self.m for i in range(self.m)]
        return nltk.translate.bleu_score.corpus_bleu(
            refs, generated, weights=weights,
            smoothing_function=smoothing.method1
        )
