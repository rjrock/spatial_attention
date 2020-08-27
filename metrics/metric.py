# metric.py

import nltk

from abc import ABC, abstractmethod
from collections import defaultdict


def count_ngrams(words, m, split=False):
    '''Return count of ngrams for tokenized words.'''
    counts = defaultdict(int)
    for k in range(1, m+1):
        for ngram in nltk.ngrams(words, k):
            counts[ngram] += 1
    return counts


class Metric(ABC):
    @abstractmethod
    def score(self, references, hypotheses):
        pass
