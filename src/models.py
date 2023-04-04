"""
    Torch-based models in use.
    
    ---

    Written & Maintained by: 
        Astromsoc
    Last Updated at:
        Apr 3, 2023
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
        self.dropout = nn.Dropout(dropout)
        # flexibly add interim linear
        self.reg_head = nn.Linear(2 + self.bert_emb.config.hidden_size, 1) if interim_linear_dim <= 0 else nn.Sequential(
            nn.Linear(2 + self.bert_emb.config.hidden_size, interim_linear_dim),
            nn.GELU(),
            nn.Linear(interim_linear_dim, 1)
        )
    
    def forward(self, ids, wlens, nsyls):
        x = self.dropout(self.bert_emb(ids)[0].mean(dim=-2))
        x = self.reg_head(torch.cat((wlens, nsyls, x), dim=-1))
        return x