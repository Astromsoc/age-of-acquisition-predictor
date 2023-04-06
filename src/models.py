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
                 dropout: float=0.3,
                 concat_wlen: bool=True,
                 concat_nsyl: bool=True):
        super().__init__()
        # archiving
        self.configs = {
            'model_name': model_name,
            'interim_linear_dim': interim_linear_dim,
            'dropout': dropout,
        }
        self.concat_wlen = concat_wlen
        self.concat_nsyl = concat_nsyl
        # bert 
        self.bert_emb = BertModel.from_pretrained(model_name)
        # dropout
        self.dropout = dropout
        # flexibly add interim linear
        extra_ncount = int(concat_wlen) + int(concat_nsyl)
        self.reg_head = nn.Linear(extra_ncount + self.bert_emb.config.hidden_size, 1) if interim_linear_dim <= 0 else nn.Sequential(
            nn.Linear(extra_ncount + self.bert_emb.config.hidden_size, interim_linear_dim),
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
        # obtain embeddings
        x = self.locked_dropout(self.bert_emb(ids)[0].sum(dim=-2))
        # concat embeddings & additional attributes in one linear
        if self.concat_nsyl and self.concat_wlen: x = torch.cat((wlens, nsyls, x), dim=-1)
        elif self.concat_nsyl: x = torch.cat((nsyls, x), dim=-1)
        elif self.concat_wlen: x = torch.cat((wlens, x), dim=-1)
        # go through the regression head
        x = self.reg_head(x)
        return x



class dualReg(nn.Module):
    def __init__(self,
                 model_name: str='bert-base-uncased',
                 interim_linear_dim: int=1024,
                 dropout: float=0.3,
                 concat_wlen: bool=True,
                 concat_nsyl: bool=True):
        super().__init__()
        # archiving
        self.configs = {
            'model_name': model_name,
            'interim_linear_dim': interim_linear_dim,
            'dropout': dropout,
        }
        self.concat_wlen = concat_wlen
        self.concat_nsyl = concat_nsyl
        # at least one side attribute shall be considered
        extra_ncount = int(concat_wlen) + int(concat_nsyl)
        assert extra_ncount >= 1
        # bert 
        self.bert_emb = BertModel.from_pretrained(model_name)
        # dropout
        self.dropout = dropout
        # must add interim linear "to summarize emb layer"
        self.emb_reg = nn.Sequential(
            nn.Linear(self.bert_emb.config.hidden_size, interim_linear_dim),
            nn.GELU()
        )
        self.reg_head = nn.Linear(extra_ncount + interim_linear_dim, 1)
    

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
        # embedding + linear
        x = self.locked_dropout(self.bert_emb(ids)[0].mean(dim=-2))
        x = self.emb_reg(x)
        # concat embeddings & additional attributes in one linear
        if self.concat_nsyl and self.concat_wlen: x = torch.cat((wlens, nsyls, x), dim=-1)
        elif self.concat_nsyl: x = torch.cat((nsyls, x), dim=-1)
        elif self.concat_wlen: x = torch.cat((wlens, x), dim=-1)
        # regression head
        x = self.reg_head(x)
        return x




"""
    Mapping of model classes from config specification strings.
"""

ChooseYourModel = {
    'early-fused': easyReg,
    'later-fused': dualReg
}
