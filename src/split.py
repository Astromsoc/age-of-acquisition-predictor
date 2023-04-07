"""
    Script for splitting up reference aoa table age-wise.
    
    ---

    Written & Maintained by: 
        Astromsoc
    Last Updated at:
        Apr 7, 2023
"""


import os
import json
import time
import argparse
import syllables
import numpy as np
import pandas as pd
from transformers import BertTokenizer

from src.utils import CharacterTokenizer



# ratio of train/val/test after splitting
RATIOS = [0.8, 0.1, 0.1]
# the column name of which reference age to take from
#       options: ['AoAtestbased', 'AoArating]
AGECOL = 'AoAtestbased'
CHR2IDX_FILEPATH = 'data/chr2idx.txt'



def split_dataset(filepath: str, tokenizer,
                  age_col: str='AoAtestbased', 
                  ratios: list=[0.8, 0.1, 0.1]):
    """
        Separate input vocabulary into training/val/test subsets age-wise.

        Args:
            filepath (str): filepath to the original csv file
            tokenizer (BertTokenizer): bert tokenizer instance
            age_col (str): column name to acquire the age from
            ratios (list): the ratios of each subset
    """
    folder = os.path.dirname(filepath)
    subnames = ['train', 'val', 'test']
    anticlash_suffix = ''
    for i, subname in enumerate(subnames):
        if os.path.exists(os.path.join(folder, f"aoapred-{subname}.json")):
            anticlash_suffix = f"-{time.strftime('%m%d%y')}"
            break
    train, val, test = dict(), dict(), dict()
    # load original csv into dataframe
    df = pd.read_csv(filepath)
    # age wise split
    for age in df[age_col].unique():
        if np.isnan(age): continue
        words_of_age = np.random.permutation(df[df[age_col] == age]['WORD'].tolist())
        total_num = len(words_of_age)
        val_num, test_num = int(total_num * ratios[1]), int(total_num * ratios[2])
        train_num = total_num - val_num - test_num
        age = float(age)
        train[age] = list(words_of_age[:train_num])
        val[age] = list(words_of_age[train_num:train_num + val_num])
        test[age] = list(words_of_age[-test_num:])
    # shuffle again and save the splits under original input folder
    subsets = [train, val, test]
    final_filepaths = list()
    for i, subset in enumerate(subsets):
        output_filepath = os.path.join(folder, f"aoapred-{subnames[i]}{anticlash_suffix}.json")
        final_filepaths.append(output_filepath)
        subset = {w: age for age, ws in subset.items() for w in ws}
        sub_jsons = list()
        for word, age in np.random.permutation(list(subset.items())):
            sub_jsons.append({'word': word, 
                              'age': float(age), 
                              'len': len(word),
                              'syllables': syllables.estimate(word),
                              'word_tokens': dict(tokenizer(word))})
        json.dump(sub_jsons, open(output_filepath, 'w'), indent=4)
        print(f"[** FILE SAVED **] Subset [{subnames[i]}] saved to [{output_filepath}].")
    return final_filepaths




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Splitting original aoa csv file age-wise.")
    
    parser.add_argument(
        '--file', '-f', type=str,
        default='data/aoa-data.csv',
        help='(str) Filepath to the original aoa csv file.'
    )
    parser.add_argument(
        '--tokenizer', '-t', type=str,
        default='bert-base-uncased',
        choices=['bert-base-uncased', 'character'],
        help='(str) pretrained tokenizer to tokenize input words/phrases.'
    )

    args = parser.parse_args()
    assert (os.path.exists(args.file), 
            f"[** FILE NOT EXISTED **] Check filepath [{args.file}] is correct.")
    
    # load tokenizer
    TOKENIZER = (CharacterTokenizer(CHR2IDX_FILEPATH) if args.tokenizer == 'character' 
                 else BertTokenizer.from_pretrained(args.tokenizer))

    # split the input csv file
    split_dataset(filepath=args.file, 
                  tokenizer=TOKENIZER, 
                  age_col=AGECOL, 
                  ratios=RATIOS)
    