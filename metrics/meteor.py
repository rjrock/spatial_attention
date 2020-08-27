'''meteor.py

Use the meteor metric as defined by nltk to evaluate machine translation
from image to caption.
'''

import nltk

from metrics.metric import Metric


class Meteor(Metric):
    def __init__(self):
        pass

    def score(self, refs, generated):
        n = len(refs)
        score = 0
        meteor_score = nltk.translate.meteor_score.meteor_score
        for i in range(n):
            # nltk meteor expects strings as input
            string_refs = [' '.join(refs[i][j]) for j in range(len(refs[i]))]
            string_generated = ' '.join(generated[i])
            score += meteor_score(string_refs, string_generated)
        return score / n
