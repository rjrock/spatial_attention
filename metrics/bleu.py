'''bleu.py'''

import nltk

from metrics.metric import Metric


class Bleu(Metric):
    '''Calculate the mean-weighted Bleu score of a corpus of reference
    sentences and test sentences for Bleu-1 to Bleu-m through nltk. The
    default, and standard, is m=4, in which case the score is the
    average of the Bleu-1, Bleu-2, Bleu-3 and Bleu-4 scores.

    The nltk code can be seen here,
    https://www.nltk.org/_modules/nltk/translate/bleu_score.html
    '''

    def __init__(self, m=4):
        self.m = m

    def score(self, refs, generated):
        smoothing = nltk.translate.bleu_score.SmoothingFunction()
        weights = [1/self.m for i in range(self.m)]
        return nltk.translate.bleu_score.corpus_bleu(
            refs, generated, weights=weights,
            smoothing_function=smoothing.method1
        )
