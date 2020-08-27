'''CIDEr-D implementation based on
https://github.com/vrama91/coco-caption.

Of note, the reference code does not faitfhully calculate term
frequency.

See cider.pdf for an explanation of what this file does.
'''

import nltk
import numpy as np

import utils.storage as storage

from collections import defaultdict
from tqdm import tqdm

from metrics.metric import Metric, count_ngrams


def extract_document_frequency(data):
    vocab = storage.load_vocab()
    captions = []
    for image_captions in tqdm(list(data.values())):
        for caption in image_captions:
            caption = nltk.tokenize.word_tokenize(caption.lower())
            caption.append(vocab.end)
            captions.append(caption)
    corpus = compute_document_frequency(captions, m=4)
    return corpus


def compute_document_frequency(refs, m):
    gram_refs = [count_ngrams(ref, m) for ref in refs]
    document_frequency = defaultdict(int)
    for gram_ref in gram_refs:
        for ngram in set([ngram for (ngram, count) in gram_ref.items()]):
            document_frequency[ngram] += 1
    corpus = {'df': document_frequency,
              'num_images': len(refs)}
    return corpus


class Cider(Metric):
    def __init__(self, document_frequency=None, reference_length=None, m=4):
        self.df = document_frequency
        self.num_images = reference_length
        self.m = m
        self.ns = range(1, m+1)

    def gramify(self, string):
        return count_ngrams(string, self.m)

    def score(self, all_refs, tests):
        '''tests and all_refs should be tokenized.'''
        scores = np.zeros(len(tests))

        for i in range(len(tests)):
            test = tests[i]
            refs = all_refs[i]
            tfidf_test = self.cap2tfidf(test)
            ngram_scores = np.zeros(self.m)
            for ref in refs:
                tfidf_ref = self.cap2tfidf(ref)
                δ = len(test) - len(ref)
                ngram_scores += self.similarity(tfidf_test, tfidf_ref, δ)
            # Multiply by 10 for magnitude similar to Bleu
            scores[i] = (10/len(refs)) * ngram_scores.mean()
        return scores.mean()

    def similarity(self, tfidf_hyp, tfidf_ref, δ, σ=6):
        '''Compute equation 4.'''
        gaussian_penalty = np.exp(-(δ**2)/(2*σ**2))
        vals = np.zeros(self.m)
        for n in self.ns:
            val = 0
            for ngram in tfidf_hyp[n].keys():
                clipped = min(tfidf_hyp[n][ngram], tfidf_ref[n][ngram])
                val += clipped * tfidf_ref[n][ngram]
            norm_hyp = np.linalg.norm(list(tfidf_hyp[n].values()))
            norm_ref = np.linalg.norm(list(tfidf_ref[n].values()))
            if norm_hyp == 0:
                print('norm_hyp is 0, exiting')
            if norm_ref == 0:
                print('norm_ref is 0, exiting')
            if norm_hyp == 0 or norm_ref == 0:
                exit(0)
            val /= (norm_hyp * norm_ref)
            # Subtract by 1 for array indexing
            vals[n-1] = val
        vals *= gaussian_penalty
        return vals

    def cap2tfidf(self, cap):
        tfidf = {n: defaultdict(float) for n in self.ns}
        grams = self.gramify(cap)
        for ngram, count in grams.items():
            n = len(ngram)
            # The reference code does not faitfhully calculate term frequency:
            #     lengths = {n: len(cap) - (n-1) for n in self.ns}
            #     tf = count / lengths[n]
            # To match the reference code, define tf as below
            tf = count
            idf = np.log(self.num_images) - np.log(max(1, self.df[ngram]))
            tfidf[n][ngram] = tf * idf
        return tfidf
