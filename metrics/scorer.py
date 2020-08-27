'''scorer.py'''

import metrics.bleu
import metrics.cider
import metrics.meteor
import models
import utils
import utils.storage as storage

from tqdm import tqdm


class Scorer:
    '''Encapsulate the metrics used to evaluate machine translation from
    image to caption.
    '''

    def __init__(self, use_bleu=True, use_cider=True, use_meteor=True):
        self.use_bleu = use_bleu
        self.use_cider = use_cider
        self.use_meteor = use_meteor
        if use_bleu:
            self.bleu = metrics.bleu.Bleu()
            self.bleu_score = 0
        if use_cider:
            val_ngram = utils.directory.cider/'val5000_img2caps_ngram.pkl'
            df, num_images = storage.load_cider_data(val_ngram)
            self.cider = metrics.cider.Cider(df, num_images)
            self.cider_score = 0
        if use_meteor:
            self.meteor = metrics.meteor.Meteor()
            self.meteor_score = 0

    def score(self, encoder, decoder, data_loader):
        training = encoder.training
        encoder.eval()
        decoder.eval()
        model = models.Model(encoder, decoder)
        print('Evaluating metrics... ')
        for i, (images, refs) in enumerate(tqdm(data_loader)):
            captions = model.max_caption(images)
            if self.use_cider:
                self.cider_score += self.cider.score(refs, captions)
            if self.use_bleu:
                self.bleu_score += self.bleu.score(refs, captions)
            if self.use_meteor:
                self.meteor_score += self.meteor.score(refs, captions)
        self.print_score(weight=1/len(data_loader))
        if training:
            encoder.train()
            decoder.train()

    def print_score(self, weight):
        '''Print out the scores.

        weight should be 1/{the size of corpus}
        '''
        if self.use_bleu:
            self.bleu_score *= weight
            print(f'\tBleu-{self.bleu.m} score = {self.bleu_score:.4f}')
        if self.use_cider:
            self.cider_score *= weight
            print(f'\tCider score = {self.cider_score:.4f}')
        if self.use_meteor:
            self.meteor_score *= weight
            print(f'\tMeteor score = {self.meteor_score:.4f}')
