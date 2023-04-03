"""
    The script & model(s) to predict age of acquisition for words
    
    ---

    Written & Maintained by: 
        Siyu Chen (schen4@andrew.cmu.edu)
    Last Updated at:
        Apr 3, 2023
"""


import os
import json
import time
import yaml
import wandb
import argparse
import syllables
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torchsummaryX import summary
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel




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



class AoADataset(Dataset):
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



def train_collate(batch):
    ids, wlens, nsyls, ages = ([u[i] for u in batch] for i in range(4))
    # pad ids only
    ids = pad_sequence(ids, batch_first=True, padding_value=0)
    # BERT models use "0" as padding value almost for all checkpoints
    return ids, torch.tensor(wlens)[:, None], torch.tensor(nsyls)[:, None], torch.tensor(ages)[:, None]



def test_collate(batch):
    ids, wlens, nsyls = ([u[i] for u in batch] for i in range(3))
    # pad ids only
    ids = pad_sequence(ids, batch_first=True, padding_value=0)
    # BERT models use "0" as padding value almost for all checkpoints
    return ids, torch.tensor(wlens)[:, None], torch.tensor(nsyls)[:, None]



class easyReg(nn.Module):
    def __init__(self,
                 model_name: str='bert-base-uncased',
                 interim_linear_dim: int=1024):
        super().__init__()
        # archiving
        self.configs = {
            'model_name': model_name,
            'interim_linear_dim': interim_linear_dim
        }
        # bert 
        self.bert_emb = BertModel.from_pretrained(model_name)
        # flexibly add interim linear
        self.reg_head = nn.Linear(2 + self.bert_emb.config.hidden_size, 1) if interim_linear_dim <= 0 else nn.Sequential(
            nn.Linear(2 + self.bert_emb.config.hidden_size, interim_linear_dim),
            nn.GELU(),
            nn.Linear(interim_linear_dim, 1)
        )
    
    def forward(self, ids, wlens, nsyls):
        x = self.bert_emb(ids)[0].mean(dim=-2)
        x = torch.cat((wlens, nsyls, x), dim=-1)
        x = self.reg_head(x)
        return x



class Trainer:
    def __init__(self, cfgs: dict, model: nn.Module,
                 trn_loader: DataLoader, val_loader: DataLoader, device: str='cuda'):
        self.cfgs = cfgs
        self.model = model
        self.trn_loader = trn_loader
        self.val_loader = val_loader
        self.bests = {'loss': float('inf'), 'epoch': -1}
        self.best_fps = list()
        self.criterion = nn.MSELoss()
        self.epoch = 0
        self.train_losses = list()
        self.train_gradnorms = list()
        self.val_losses = list()
        self.device = device
        # init from cfgs
        self.init_from_cfgs()
        # take model to device
        self.model.to(self.device)
    

    def init_from_cfgs(self):
        self.scaler = torch.cuda.amp.GradScaler() if self.cfgs['scaler'] else None
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, **self.cfgs['scheduler']['configs']
        ) if self.cfgs['scheduler']['use'] else None
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **self.cfgs['optimizer'])


    def train_epoch(self):
        tqdm_bar = tqdm(total=len(self.trn_loader), leave=False, dynamic_ncols=True,
                        desc=f"training epoch [{self.epoch + 1:<3}]")
        train_loss_this_epoch = np.zeros((len(self.trn_loader),))
        grad_norm = np.zeros((len(self.trn_loader),))
        # switch to training mode
        self.model.train()
        for i, (ids, wlens, nsyls, ages) in enumerate(self.trn_loader):
            # take to device
            ids, wlens = ids.to(self.device), wlens.to(self.device)
            nsyls, ages = nsyls.to(self.device), ages.to(self.device)
            # obtain estimates
            pred_ages = self.model(ids, wlens, nsyls)
            # compute loss
            loss = self.criterion(pred_ages, ages)
            train_loss_this_epoch[i] = loss.item()
            # backprop & update
            if self.scaler:
                with torch.cuda.amp.autocast():
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            # compute gradient norm
            grad_norm[i] = sum([p.grad.data.detach().norm(2) for p in self.model.parameters() if p.grad]) ** 0.5
            # clear grad
            self.optimizer.zero_grad()
            # update batch bar
            tqdm_bar.set_postfix(
                progress=f"{i+1:<3}/{len(self.trn_loader)}",
                train_loss=f"{train_loss_this_epoch[i]:.6f}",
                grad_norm=f"{grad_norm[i]:6f}",
                lr=self.optimizer.param_groups[0]['lr']
            )
            tqdm_bar.update()
            # push to wandb
            if self.cfgs['wandb']['use']:
                wandb.log({"train_loss_per_batch": f"{train_loss_this_epoch[i]:.6f}",
                           "grad_norm_per_batch": f"{grad_norm[i]:6f}"})
            # update learning rate
            if self.scheduler: 
                self.scheduler.step(loss)
                if self.cfgs['wandb']['use']:
                    wandb.log({"lr_per_batch": self.optimizer.param_groups[0]['lr']})

        # clear
        del ids, wlens, nsyls, ages
        torch.cuda.empty_cache()
        tqdm_bar.close()

        return train_loss_this_epoch.mean(), grad_norm.mean()
    

    def eval_epoch(self):
        tqdm_bar = tqdm(total=len(self.val_loader), leave=False, dynamic_ncols=True,
                        desc=f"eval epoch [{self.epoch + 1:<3}]")
        val_loss_this_epoch = np.zeros((len(self.val_loader),))
        # switch to inference mode
        self.model.eval()
        with torch.inference_mode():
            for i, (ids, wlens, nsyls, ages) in enumerate(self.val_loader):
                # take to device
                ids, wlens = ids.to(self.device), wlens.to(self.device)
                nsyls, ages = nsyls.to(self.device), ages.to(self.device)
                # obtain estimates
                pred_ages = self.model(ids, wlens, nsyls)
                # compute loss
                loss = self.criterion(pred_ages, ages)
                val_loss_this_epoch[i] = loss.item()
                # update batch bar
                tqdm_bar.set_postfix(
                    progress=f"{i+1:<3}/{len(self.trn_loader)}",
                    val_loss=f"{val_loss_this_epoch[i]:.6f}"
                )
                tqdm_bar.update()
                # push to wandb
                if self.cfgs['wandb']['use']:
                    wandb.log({"val_loss_per_batch": f"{val_loss_this_epoch[i]:.6f}"})

        # clear
        del ids, wlens, nsyls, ages
        torch.cuda.empty_cache()
        tqdm_bar.close()

        return val_loss_this_epoch.mean()
    

    def train(self, expcfgs: dict):
        """
            Train the current model for what's been specified in input experiment configs.
        """

        # if finetuning: load checkpoint
        if expcfgs['finetune']['use']:
            self.load_model(expcfgs['finetune']['ckpt'])

        while self.epoch <= expcfgs['epoch']:
            train_avg_loss, train_avg_grad_norm = self.train_epoch()
            val_avg_loss = self.eval_epoch()
            # record
            self.train_losses.append(train_avg_loss)
            self.train_gradnorms.append(train_avg_grad_norm)
            self.val_losses.append(val_avg_loss)
            # push to wandb
            if self.cfgs['wandb']['use']:
                wandb.log({'train_loss_per_epoch': self.train_losses[-1],
                           'train_gradnorm_per_epoch': self.val_losses[-1],
                           'val_loss_per_epoch': self.val_losses[-1],
                           'lr_per_epoch': self.optimizer.param_groups[0]['lr']})
            # save model
            self.save_model(expcfgs['exp_folder'])
            # increment epoch by 1
            self.epoch += 1
            # save model
            self.save_model(expcfgs)

    
    def save_model(self, expcfgs: str):
        """
            Save a model checkpoint to specified experiment folder.
        """
        # check if a lower val MSE is reached or the bests are not reached
        if self.val_losses[-1] < self.bests['loss'] or len(self.best_fps) < self.cfgs['max_saved_ckpts']:
            # update best model stats
            if self.val_losses[-1] < self.bests['loss']:
                self.bests = {'loss': self.val_losses[-1], 'epoch': self.epoch}
            # sort the saved checkpoints (before reaching maximum storage)
            if len(self.best_fps) < self.cfgs['max_saved_ckpts']:
                self.best_fps = [self.best_fps[i] for i in sorted(list(range(len(self.best_fps))), lambda i: -self.val_losses[i])]
            # save checkpoint
            if len(self.best_fps) >= self.cfgs['max_saved_ckpts']:
                # delete the oldest checkpoint
                os.remove(self.best_fps.pop(0))
            # create folder if not existed
            if not os.path.exists(expcfgs['exp_folder']):
                os.makedirs(expcfgs['exp_folder'], exist_ok=True)
            # add new filepath
            self.best_fps.append(os.path.join(expcfgs['exp_folder'], f"epoch-{self.epoch}.pt"))
            # save model checkpoint
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optim_state_dict': self.optimizer.state_dict(),
                'epoch': self.epoch,
                'best_fps': self.best_fps,
                'bests': self.bests,
                'train_losses': self.train_losses,
                'train_gradnorms': self.train_gradnorms,
                'val_losses': self.val_losses,
                'configs': {'trainer': self.cfgs, 'exp': expcfgs}
            }, self.best_fps[-1])
            print(f"\n[** MODEL SAVED **] Successfully saved checkpoint to [{self.best_fps[-1]}]\n")
    

    def load_model(self, ckpt_filepath: str):
        """
            Load a model checkpoint from specified filepath.
        """
        assert (os.path.exists(ckpt_filepath), 
                f"\n[** FILE NOT EXISTED **] Can't load from [{ckpt_filepath}].\n")
        loaded = torch.load(open(ckpt_filepath, 'r'))
        print(f"\n[** MODEL LOADED **] Successfully loaded checkpoint from [{ckpt_filepath}]\n")
        # load configs
        self.cfgs = loaded['configs']['trainer']
        # init from configs
        self.init_from_cfgs()
        # other state dicts / saved attributes
        self.model.load_state_dict(loaded['model_state_dict'])
        self.optimizer.load_state_dict(loaded['optim_state_dict'])
        self.epoch = loaded['epoch']
        self.best_fps = loaded['best_fps']
        self.bests = loaded['bests']
        self.train_losses = loaded['train_losses']
        self.train_gradnorms = loaded['train_gradnorms']
        self.val_losses = loaded['val_losses']

        


"""
    Main Driver Function
"""

def main(args):
    # load configurations
    cfgs = yaml.safe_load(open(args.config))

    # fix random seeds
    np.random.seed(cfgs['seed'])

    # obtain device
    device = ('mps' if torch.backends.mps.is_available() else
              'cuda' if torch.cuda.is_available() else
              'cpu')
    print(f"\n[** DEVICE FOUND **] Now running on [{device}].\n")

    # build tokenizer
    TOKENIZER = BertTokenizer.from_pretrained(cfgs['tokenizer_name'])

    # split dataset
    if args.start_stage == 0 or (
        not os.path.exists(cfgs['aoapred_train_filepath'])
        or not os.path.exists(cfgs['aoapred_val_filepath'])
        or not os.path.exists(cfgs['aoapred_test_filepath'])
        # re-split when any of them does not exist
    ):
        subset_filepaths = split_dataset(cfgs['aoa_csv_filepath'], TOKENIZER)
        for i, subname in enumerate('train val test'.split(' ')):
            # update configs
            cfgs[f"aoapred_{subname}_filepath"] = subset_filepaths[i]
    
    # build datasets
    trainDataset = AoADataset(cfgs['aoapred_train_filepath'])
    valDataset = AoADataset(cfgs['aoapred_val_filepath'])

    # build dataloaders
    trainLoader = DataLoader(dataset=trainDataset, 
                             collate_fn=train_collate, 
                             **cfgs['train_loader'])
    valLoader = DataLoader(dataset=valDataset, 
                           shuffle=False, 
                           collate_fn=train_collate, 
                           **cfgs['val_loader'])

    # build model
    model = easyReg(**cfgs['model_configs'])

    # show model summary
    model.eval()
    ids, wlen, nsyl, _ = next(iter(trainLoader))
    print(summary(model, ids, wlen, nsyl))

    # build trainer
    trainer = Trainer(
        cfgs=cfgs['trainer_configs'], model=model, trn_loader=trainLoader,
        val_loader=valLoader, device=device
    )

    # start training
    trainer.train()
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="building & predicting age of acquisition")

    parser.add_argument(
        '--start_stage', '-ss', default=1, type=int,
        help="(int) Stage to start running the script."
    )
    parser.add_argument(
        '--end_stage', '-es', default=1, type=int,
        help="(int) Stage to end running the script."
    )
    parser.add_argument(
        '--config', '-c', default='samples/sample-aoapred-configs.yaml', type=str,
        help="(str) Filepath to the configuration."
    )

    args = parser.parse_args()
    assert (args.config.endswith((".yaml", ".yml")) and os.path.exists(args.config), 
            f"[** FILEPATH NOT EXISTED **] Can't load config file from {args.config}.")
    main(args)