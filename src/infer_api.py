"""
    Script to infer age of acquisition for random input text (API version).
    
    ---

    Written & Maintained by: 
        Astromsoc
    Last Updated at:
        Apr 6, 2023
"""


import re
import os
import wandb
import argparse
import syllables
import numpy as np
from tqdm import tqdm
from ruamel.yaml import YAML
yaml = YAML(typ='safe')

import torch
from transformers import BertTokenizer

from src.utils import *
from src.models import *
from src.split import split_dataset


FILTERED_REGEX = re.compile(r"[^a-z\.' -]*")
TRAILING_PERIOD_REGEX = re.compile(r"^([a-z' -]+)(\.)?$")
AGE_RANGE_REGEX = re.compile(r"\((\d+),( )?(\d+)\)")



class InferrerAPI:
    def __init__(self, 
                 ckpt: str, 
                 tokenizer_name: str='bert-base-uncased',
                 device: str='cuda'):
        self.ckpt = ckpt
        self.device = device
        # init from cfgs
        self.exp_folder = os.path.dirname(self.ckpt)
        self.mcfg_filepath = os.path.join(self.exp_folder, 'configs.yaml')
        self.reload_ckpt()
    

    def reload_ckpt(self):
        # load checkpoint
        assert os.path.exists(self.ckpt), f"\n[** FILE NOT EXISTED **] Can't load from [{self.ckpt}].\n"
        loaded = torch.load(self.ckpt, map_location=torch.device(self.device))
        print(f"\n[** MODEL LOADED **] Successfully loaded checkpoint from [{self.ckpt}]\n")
        # build hollow model given the copied model configs
        self.modelcfgs = ParamsObject(yaml.load(open(self.mcfg_filepath, 'r'))).model
        self.model = ChooseYourModel[self.modelcfgs.choice](**self.modelcfgs.configs.__dict__)
        # load model state dict
        self.model.load_state_dict(loaded['model_state_dict'])
        # take model to device
        self.model.to(self.device)
        # load scaler
        self.scaler = torch.cuda.amp.GradScaler() if loaded['configs']['trainer'].scaler else None
        # build tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            loaded['configs'].get('tokenizer', 'bert-base-uncased')
        )

    def infer_text(self, text: str):
        # tokenize
        tokens = re.sub(r"[^a-z/.' -]*", "", text.lower()).split(' ')

        # record each token's output result
        tokens2ages = dict()
        for token in tokens:
            # remove trailing period
            token = re.match(TRAILING_PERIOD_REGEX, token).group(1)
            # obtain all inputs to model
            token_ids = dict(self.tokenizer(token))['input_ids']
            nsyl, wlen = syllables.estimate(token), len(token)
            tokens2ages[token] = self.model(
                torch.tensor(token_ids, device=self.device).unsqueeze(0),
                torch.tensor(wlen, device=self.device).view(1, 1), 
                torch.tensor(nsyl, device=self.device).view(1, 1)
            ).item()
        
        return np.mean(list(tokens2ages.values())), tokens2ages


"""
    Main Driver Function
"""

def main(args):

    # load configurations
    cfgs = ParamsObject(yaml.load(open(args.config, 'r')))

    # obtain device
    device = ('mps' if torch.backends.mps.is_available() else
              'cuda' if torch.cuda.is_available() else
              'cpu')
    print(f"\n[** DEVICE FOUND **] Now running on [{device}].\n")

    # build inferrer
    inferrer = InferrerAPI(ckpt=cfgs.ckpt, device=device)


    while True:

        # input random text
        age_range_str = input("\n\nPlease input the age group, formatted as (low, high):\n")
        m = re.match(AGE_RANGE_REGEX, age_range_str)
        ages = (float(m.group(1)), float(m.group(3)))
        text = input("\n\nPlease input the text for age prediction:\n")
        # inferrence
        avg_age, specifics = inferrer.infer_text(text)
        # display
        print_textpred_case(text=text, avg_age=avg_age, tokens2ages=specifics, age_range=ages)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Predicting age of acquisition.")
    parser.add_argument(
        '--config', '-c', default='cfg/sample-infer-api-configs.yaml', type=str,
        help="(str) Filepath to the configuration."
    )

    args = parser.parse_args()
    assert args.config.endswith((".yaml", ".yml")) and os.path.exists(args.config), f"[** FILEPATH NOT EXISTED **] Can't load config file from {args.config}."

    main(args)