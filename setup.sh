#!/bin/bash

# [1] build a conda env for experiments
yes | conda create -n aoapred python=3.8 && conda activate aoapred

# [2] install dependencies
pip install -r requirements.txt
yes | conda install tmux

# [3] login to wandb account
wandb login