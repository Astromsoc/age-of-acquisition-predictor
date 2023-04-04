"""
    Torch-based models in use.
    
    ---

    Written & Maintained by: 
        Astromsoc
    Last Updated at:
        Apr 4, 2023
"""


import torch
import torch.nn as nn
from transformers import BertModel



class easyReg(nn.Module):
    def __init__(self,
                 model_name: str='bert-base-uncased',
                 interim_linear_dim: int=1024,
                 dropout: float=0.3):
        super().__init__()
        # archiving
        self.configs = {
            'model_name': model_name,
            'interim_linear_dim': interim_linear_dim,
            'dropout': dropout
        }
        # bert 
        self.bert_emb = BertModel.from_pretrained(model_name)
        # dropout
        self.dropout = dropout
        # flexibly add interim linear
        self.reg_head = nn.Linear(2 + self.bert_emb.config.hidden_size, 1) if interim_linear_dim <= 0 else nn.Sequential(
            nn.Linear(2 + self.bert_emb.config.hidden_size, interim_linear_dim),
            nn.GELU(),
            nn.Linear(interim_linear_dim, 1)
        )
    

    def locked_dropout(self, x):
        """
            keep the same dropout for the entire batch
        """
        if (not self.training) or (not self.dropout):
            return x
        mask = x.new_empty(1, x.size(1)).bernoulli_(1 - self.dropout)
        mask = mask.div_(1 - self.dropout).expand_as(x)
        return x * mask

    
    def forward(self, ids, wlens, nsyls):
        x = self.locked_dropout(self.bert_emb(ids)[0].mean(dim=-2))
        x = self.reg_head(torch.cat((wlens, nsyls, x), dim=-1))
        return x