'''data_loader.py'''

import json
import nltk
import os
import torch
import torch.utils.data as data

from PIL import Image
from torchvision import transforms


class TrainDataset(data.Dataset):
    def __init__(self, root, cap2img_file, img2caps_file, vocab, transform):
        super(data.Dataset, self)
        self.root      = root
        self.cap2img   = json.load(open(cap2img_file))
        self.img2caps  = json.load(open(img2caps_file))
        self.vocab     = vocab
        self.transform = transform

    def one_hot(self, string, start=True, end=True):
        tokens = nltk.tokenize.word_tokenize(string.lower())
        token_idxs = []
        if start:
            token_idxs.append(self.vocab.start_idx)
        token_idxs.extend([self.vocab(token) for token in tokens])
        if end:
            token_idxs.append(self.vocab.end_idx)
        return token_idxs

    def __getitem__(self, idx):
        caption, imagename = self.cap2img[idx].values()
        image = Image.open(os.path.join(self.root, imagename)).convert('RGB')
        image = self.transform(image)
        refs = [nltk.tokenize.word_tokenize(ref.lower()) + [self.vocab.end]
                for ref in self.img2caps[imagename]]
        token_idxs = self.one_hot(caption)
        target = torch.Tensor(token_idxs)
        return image, target, refs

    def __len__(self):
        return len(self.cap2img)

    def collate(self, data):
        '''Sort the data from longest caption-lengths to shortest
        caption-lengths. This is the format required by
        pack_padded_sequence.

        Stack the images and captions into batches.

        The lengths of the captions are decreased by 1, since the inputs
        don't include <end> and the targets don't include <start>.

        The refs are reference captions and are used for calculating
        metrics.
        '''
        data.sort(key=lambda x: len(x[1]), reverse=True)
        images, targets, refs = zip(*data)
        batch_size = len(targets)
        images = torch.stack(images, 0)
        max_caption_length = max(len(target) for target in targets)
        padded_inputs  = torch.zeros(batch_size, max_caption_length).long()
        padded_targets = torch.zeros(batch_size, max_caption_length).long()
        lengths = []
        for i, target in enumerate(targets):
            length = len(target) - 1
            padded_inputs[i, :length]  = target[:-1]
            padded_targets[i, :length] = target[1:]
            lengths.append(length)
        return images, padded_inputs, padded_targets, lengths, refs


class ValidationDataset(data.Dataset):
    '''Load from img2caps as opposed to TrainData that loads from cap2img.'''

    def __init__(self, root, img2caps_file, vocab, transform):
        super(data.Dataset, self)
        self.root      = root
        self.img2caps  = json.load(open(img2caps_file))
        self.imagenames = list(self.img2caps.keys())
        self.vocab     = vocab
        self.transform = transform

    def __getitem__(self, idx):
        imagename = self.imagenames[idx]
        image = Image.open(os.path.join(self.root, imagename)).convert('RGB')
        image = self.transform(image)
        refs = [nltk.tokenize.word_tokenize(ref.lower()) + [self.vocab.end]
                for ref in self.img2caps[imagename]]
        return image, refs

    def __len__(self):
        return len(self.imagenames)

    def collate(self, data):
        images, refs = zip(*data)
        images = torch.stack(images, 0)
        return images, refs


def validation_loader(root, img2caps_file, vocab, batch_size, num_workers,
                      transform=None):
    if not transform:
        transform = transforms.Compose([
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
    coco = ValidationDataset(root, img2caps_file, vocab, transform)
    shuffle = False
    data_loader = torch.utils.data.DataLoader(
        dataset=coco, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=coco.collate
    )
    return data_loader


def train_loader(root, cap2img_file, img2caps_file, vocab, batch_size, shuffle,
                 num_workers, transform=None):
    # Do I need to pass in shuffle?
    if not transform:
        transform = transforms.Compose([
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
    coco = TrainDataset(root, cap2img_file, img2caps_file, vocab, transform)
    data_loader = torch.utils.data.DataLoader(
        dataset=coco, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=coco.collate
    )
    return data_loader
