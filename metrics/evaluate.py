'''evaluate.py'''

import matplotlib.pyplot as plt

import metrics.bleu
import models
import utils.image as image


def evaluate(encoder, decoder, data_loader):
    bleu = metrics.bleu.Bleu()
    training = encoder.training
    encoder.eval()
    decoder.eval()
    model = models.Model(encoder, decoder)
    bleu_score = 0
    for i, (images, refs) in enumerate(data_loader):
        captions = model.max_caption(images)
        bleu_score += bleu.score(refs, captions)
    bleu_score /= len(data_loader)
    print(f'\tBleu score = {bleu_score:.4f}')
    if training:
        encoder.train()
        decoder.train()
