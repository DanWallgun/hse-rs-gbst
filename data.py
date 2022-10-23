import datasets
import re
from string import ascii_lowercase

import torch
import sentencepiece as spm


def load_dataset(name):
    datasets.logging.set_verbosity_error()
    return datasets.load_dataset(name)


def normalize_text(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z ]", "", text)
    return text


class CharTextEncoder():
    def __init__(self):
        alphabet = list(ascii_lowercase + ' ' + '\0')
        self.ind2char = {k: v for k, v in enumerate(sorted(alphabet))}
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.char2ind)
    
    def encode(self, text):
        text = normalize_text(text)
        return [self.char2ind[char] for char in text]


class SPMTextEncoder():
    def __init__(self, filename='en.wiki.bpe.vs1000.model'):
        self.model = spm.SentencePieceProcessor(filename)
    
    def __len__(self):
        return self.model.GetPieceSize()
    
    def encode(self, text):
        text = normalize_text(text)
        return self.model.EncodeAsIds(text)


def collate_batch(batch, text_encoder, device):
    label_list, text_list, lengths = [], [], []

    for item in batch:
        label_list.append(item['label'])
        processed_text = torch.tensor(text_encoder.encode(item['text']), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))
    
    label_list = torch.tensor(label_list, dtype=torch.int64)
    lengths = torch.tensor(lengths)
    text_padded = torch.nn.utils.rnn.pad_sequence(text_list, batch_first=True)

    return {
        'labels': label_list.to(device),
        'text': [normalize_text(item['text']) for item in batch],
        'text_padded': text_padded.to(device),
        'lengths': lengths.to(device),
    }
