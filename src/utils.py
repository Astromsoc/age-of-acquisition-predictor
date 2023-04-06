"""
    Utility classes / functions in use.
    
    ---

    Written & Maintained by: 
        Astromsoc
    Last Updated at:
        Apr 6, 2023
"""


import json
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence



class ParamsObject(object):
    """
        Convert yaml dictionary into object for cleaner calls 
    """
    def __init__(self, cfg_dict: dict):
        self.__dict__.update(cfg_dict)
        for k, v in self.__dict__.items():
            if isinstance(v, dict): self.__dict__.update({k: ParamsObject(v)})



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



class AoATestDatasetWordOnly(Dataset):
    def __init__(self, filepath: str, tokenizer_name: str):
        super().__init__()
        # archiving
        self.filepath = filepath
        self.tokenizer_name = tokenizer_name
        # only loading words
        self.words = [l.strip() for l in open(self.filepath, 'r')]
        # build tokenizer & convert to token_ids
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
        self.token_dicts = [self.tokenizer(w) for w in self.words]
        self.word_input_ids = [torch.tensor(u['input_ids']) for u in self.token_dicts]

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        return self.word_input_ids[index]



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



def print_textpred_case(text: str, 
                        avg_age: float, 
                        tokens2ages: dict,
                        age_range: tuple):
    """
        Print the prediction results with highlights on potentially difficult words.
    """
    # showing inputs
    print(f"\n---\nOriginal Story:\n{text}")
    print("---\nAnticipated Age Group: ({}, {})".format(*age_range))
    # showing pred results
    print(f"---\nOverall Age: {avg_age:.2f}\n---")
    print(f"Broken-downs:")
    for t, age in tokens2ages.items():
        useHighlight = True if age > age_range[1] else False
        print(f"{t:>20} | {age:.2f}" + ("**" if useHighlight else ""))
    print(f"---\n\n")