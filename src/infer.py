"""
    Script to infer age of acquisition for words.
    
    ---

    Written & Maintained by: 
        Astromsoc
    Last Updated at:
        Apr 11, 2023
"""


import os
import argparse
import numpy as np
from tqdm import tqdm
from ruamel.yaml import YAML
yaml = YAML(typ='safe')

import torch
from torch.utils.data import DataLoader

from src.utils import *
from src.models import *



class Inferer:
    def __init__(self, ckpt: str, device: str='cuda'):
        self.ckpt = ckpt
        self.device = device
        self.criterion = nn.MSELoss()
        # init from cfgs
        self.exp_folder = os.path.dirname(self.ckpt)
        self.mcfg_filepath = os.path.join(self.exp_folder, 'model-configs.yaml')
        self.reload_ckpt()
    

    def reload_ckpt(self):
        # load checkpoint
        assert os.path.exists(self.ckpt), f"\n[** FILE NOT EXISTED **] Can't load from [{self.ckpt}].\n"
        loaded = torch.load(self.ckpt, map_location=torch.device(self.device))
        print(f"\n[** MODEL LOADED **] Successfully loaded checkpoint from [{self.ckpt}]\n")

        # build hollow model given the copied model configs
        self.modelcfgs = ParamsObject(yaml.load(open(self.mcfg_filepath, 'r')))
        self.model = ChooseYourModel[self.modelcfgs.choice](**self.modelcfgs.configs.__dict__)
        # load model state dict
        self.model.load_state_dict(loaded['model_state_dict'])
        # take model to device
        self.model.to(self.device)

        # load scaler
        self.scaler = torch.cuda.amp.GradScaler() if loaded['configs']['trainer']['scaler'] else None


    def infer(self, tst_loader: DataLoader, with_labels: bool=False):
        tqdm_bar = tqdm(total=len(tst_loader), leave=False, dynamic_ncols=True,
                        desc=f"inferring...")
        inferred_outputs = list()
        if with_labels:
            test_loss_this_epoch = np.zeros((len(tst_loader),))
            test_mae_this_epoch = np.zeros((len(tst_loader),))
        # switch to inference mode
        self.model.eval()
        with torch.inference_mode():
            for i, bundle in enumerate(tst_loader):
                # it's possible that the test set has labels too -- similar to val set
                if with_labels: ids, wlens, nsyls, ages = bundle
                else: ids, wlens, nsyls = bundle
                # take to device
                ids, wlens, nsyls = ids.to(self.device), wlens.to(self.device), nsyls.to(self.device)
                if with_labels: ages = ages.to(self.device)
                # obtain estimates
                pred_ages = self.model(ids, wlens, nsyls)
                # compute mae & loss
                if with_labels: 
                    test_mae_this_epoch[i] = (pred_ages - ages).norm(1) / len(pred_ages)
                    test_loss_this_epoch[i] = self.criterion(pred_ages, ages).item()
                # add predictions to output list
                inferred_outputs += pred_ages.tolist()
                # update batch bar
                tqdm_bar.set_postfix(
                    test_loss=f"{test_loss_this_epoch[i]:.6f}",
                    test_mae=f"{test_mae_this_epoch[i]:.6f}"
                )
                tqdm_bar.update()
        # clear
        del ids, wlens, nsyls, ages
        torch.cuda.empty_cache()
        tqdm_bar.close()

        return ((inferred_outputs, test_loss_this_epoch.mean(), test_mae_this_epoch.mean()) 
                if with_labels else inferred_outputs)
        
    

"""
    Main Driver Function
"""

def main(args):

    # load configurations
    cfgs = ParamsObject(yaml.load(open(args.config, 'r')))
    addGoldens = cfgs.consider_labels

    # obtain device
    device = ('mps' if torch.backends.mps.is_available() else
              'cuda' if torch.cuda.is_available() else
              'cpu')
    print(f"\n[** DEVICE FOUND **] Now running on [{device}].\n")

    
    # build datasets
    testDataset = (AoATrainDataset(cfgs.aoapred_test_filepath,
                                   pad_idx=cfgs.pad_idx) if addGoldens else
                   AoATestDataset(cfgs.aoapred_test_filepath, pad_idx=cfgs.pad_idx))

    # build dataloader
    testLoader = DataLoader(dataset=testDataset, shuffle=False, 
                            collate_fn=train_collate if addGoldens else test_collate, 
                            **cfgs.test_loader.__dict__)

    # build inferer
    inferer = Inferer(ckpt=cfgs.ckpt, device=device)

    # infer all cases
    preds = inferer.infer(testLoader, with_labels=addGoldens)
    if addGoldens: preds, test_avg_loss, test_avg_mae = preds

    # record the results
    output_filepath = os.path.join(
        inferer.exp_folder, 
        os.path.basename(cfgs.aoapred_test_filepath).replace('.json', '-inferred.json')
    )
    for i, pred in enumerate(preds):
        preds[i] = {'word': testDataset.words[i], 'pred_age': pred[0]}
        if addGoldens:  preds[i]['gold_age'] = testDataset.ages[i]
    # save the result under exp folder
    if addGoldens: preds.append({'test_avg_loss': test_avg_loss,
                                 'test_avg_mae': test_avg_mae})
    json.dump(preds, open(output_filepath, 'w'), indent=4) 
    print(f"\n[** PRED FILE SAVED **] Predictions successfully saved to [{output_filepath}].\n")
    
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Predicting age of acquisition.")
    parser.add_argument(
        '--config', '-c', default='cfg/sample-infer-configs.yaml', type=str,
        help="(str) Filepath to the configuration."
    )

    args = parser.parse_args()
    assert (args.config.endswith((".yaml", ".yml")) and os.path.exists(args.config), 
            f"[** FILEPATH NOT EXISTED **] Can't load config file from {args.config}.")

    main(args)
