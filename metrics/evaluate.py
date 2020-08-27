'''evaluate.py'''

import metrics.bleu
import models

from tqdm import tqdm


def setup():
    pass


def evaluate(encoder, decoder, data_loader):
    bleu = metrics.bleu.Bleu()
    training = encoder.training
    encoder.eval()
    decoder.eval()
    model = models.Model(encoder, decoder)
    bleu_score = 0
    print('Evaluating metrics... ')
    # TODO: Use cider
    for i, (images, refs) in enumerate(tqdm(data_loader)):
        captions = model.max_caption(images)
        bleu_score += bleu.score(refs, captions)
    bleu_score /= len(data_loader)
    print(f'\tBleu-{bleu.m} score = {bleu_score:.4f}')
    if training:
        encoder.train()
        decoder.train()
