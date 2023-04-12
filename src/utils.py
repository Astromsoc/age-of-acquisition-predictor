"""
    Utility classes / functions in use.
    
    ---

    Written & Maintained by: 
        Astromsoc
    Last Updated at:
        Apr 11, 2023
"""


import re
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
    def __init__(self, filepath: str, pad_idx: int):
        super().__init__()
        # archiving
        self.filepath = filepath
        self.pad_idx = pad_idx
        # only loading 3 attributes
        self.raws  = json.load(open(filepath, 'r'))
        self.words = [u['word'] for u in self.raws]
        self.ages  = [u['age'] for u in self.raws]
        self.lens  = [u['len'] for u in self.raws]
        self.syllables = [u['syllables'] for u in self.raws]
        self.word_input_ids = [torch.tensor(u['word_tokens']['input_ids']) for u in self.raws]


    def __len__(self):
        return len(self.words)


    def __getitem__(self, index):
        return self.word_input_ids[index], self.lens[index], self.syllables[index], self.ages[index]


    def collate_fn(self, batch):
        ids, wlens, nsyls, ages = ([u[i] for u in batch] for i in range(4))
        # pad ids only
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad_idx)
        # BERT models use "0" as padding value almost for all checkpoints
        return (ids, torch.tensor(wlens)[:, None], 
                     torch.tensor(nsyls)[:, None], 
                     torch.tensor(ages)[:, None])



class AoATestDataset(Dataset):
    def __init__(self, filepath: str, pad_idx: int):
        super().__init__()
        # archiving
        self.filepath = filepath
        self.pad_idx = pad_idx
        # only loading 3 attributes
        self.raws  = json.load(open(filepath, 'r'))
        self.words = [u['word'] for u in self.raws]
        self.lens  = [u['len'] for u in self.raws]
        self.syllables = [u['syllables'] for u in self.raws]
        self.word_input_ids = [torch.tensor(u['word_tokens']['input_ids']) for u in self.raws]


    def __len__(self):
        return len(self.words)


    def __getitem__(self, index):
        return self.word_input_ids[index], self.lens[index], self.syllables[index]
    

    def collate_fn(batch):
        ids, wlens, nsyls = ([u[i] for u in batch] for i in range(3))
        # pad ids only
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad_idx)
        # BERT models use "0" as padding value almost for all checkpoints
        return ids, torch.tensor(wlens)[:, None], torch.tensor(nsyls)[:, None]




class AoATestDatasetWordOnly(Dataset):
    def __init__(self, filepath: str, tokenizer_name: str, pad_idx: int):
        super().__init__()
        # archiving
        self.filepath = filepath
        self.tokenizer_name = tokenizer_name
        self.pad_idx = pad_idx
        # only loading words
        self.words = [l.strip() for l in open(self.filepath, 'r')]
        # build tokenizer & convert to token_ids
        self.tokenizer      = BertTokenizer.from_pretrained(self.tokenizer_name)
        self.token_dicts    = [self.tokenizer(w) for w in self.words]
        self.word_input_ids = [torch.tensor(u['input_ids']) for u in self.token_dicts]


    def __len__(self):
        return len(self.words)


    def __getitem__(self, index):
        return self.word_input_ids[index]


    def collate_fn(batch):
        ids, wlens, nsyls = ([u[i] for u in batch] for i in range(3))
        # pad ids only
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad_idx)
        # BERT models use "0" as padding value almost for all checkpoints
        return ids, torch.tensor(wlens)[:, None], torch.tensor(nsyls)[:, None]



class CharacterTokenizer:

    """
        Per-character tokenizer that convert words into list of character indices.
        (mainly used in pretraining embeddings)
    """

    CHRLINE_REGEX = re.compile(r"(.+)\t([\d]+)")

    def __init__(self, chr2idx_filepath: str):
        # archiving
        self.filepath = chr2idx_filepath
        # load dictionary
        self.chr2idx = {c: idx for l in open(self.filepath, 'r')
                               for (c, idx) in [self.parse_chr2idx_line(l)]}
        self.unk_idx = self.chr2idx['<unk>']
    

    def __call__(self, word: str):
        return self.tokenize(word)
    

    def tokenize(self, word: str):
        """
            similar to the style of BertPretrainedTokenizer (for compatibility)
        """
        return {'input_ids': [self.chr2idx.get(c, self.unk_idx) for c in list(word)]}


    def parse_chr2idx_line(self, chr2idx: str):
        m = re.match(self.CHRLINE_REGEX, chr2idx)
        return m.group(1), int(m.group(2))



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
