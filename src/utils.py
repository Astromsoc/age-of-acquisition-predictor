"""
    Utility classes / functions in use.
    
    ---

    Written & Maintained by: 
        Astromsoc
    Last Updated at:
        Apr 3, 2023
"""


import json
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence



class AoATrainDataset(Dataset):
    def __init__(self, filepath: str):
        super().__init__()
        # archiving
        self.filepath = filepath
        # only loading 3 attributes
        self.raws = json.load(open(filepath, 'r'))
        self.words = [u['word'] for u in self.raws]
        self.ages = [u['age'] for u in self.raws]
        self.lens = [u['len'] for u in self.raws]
        self.syllables = [u['syllables'] for u in self.raws]
        self.word_input_ids = [torch.tensor(u['word_tokens']['input_ids']) for u in self.raws]

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        return self.word_input_ids[index], self.lens[index], self.syllables[index], self.ages[index]



class AoATestDataset(Dataset):
    def __init__(self, filepath: str):
        super().__init__()
        # archiving
        self.filepath = filepath
        # only loading 3 attributes
        self.raws = json.load(open(filepath, 'r'))
        self.words = [u['word'] for u in self.raws]
        self.lens = [u['len'] for u in self.raws]
        self.syllables = [u['syllables'] for u in self.raws]
        self.word_input_ids = [torch.tensor(u['word_tokens']['input_ids']) for u in self.raws]

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        return self.word_input_ids[index], self.lens[index], self.syllables[index]



def train_collate(batch):
    ids, wlens, nsyls, ages = ([u[i] for u in batch] for i in range(4))
    # pad ids only
    ids = pad_sequence(ids, batch_first=True, padding_value=0)
    # BERT models use "0" as padding value almost for all checkpoints
    return (ids, 
            torch.tensor(wlens)[:, None], 
            torch.tensor(nsyls)[:, None], 
            torch.tensor(ages)[:, None])



def test_collate(batch):
    ids, wlens, nsyls = ([u[i] for u in batch] for i in range(3))
    # pad ids only
    ids = pad_sequence(ids, batch_first=True, padding_value=0)
    # BERT models use "0" as padding value almost for all checkpoints
    return (ids, 
            torch.tensor(wlens)[:, None], 
            torch.tensor(nsyls)[:, None])
