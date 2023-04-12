"""
    Torch-based models in use.
    
    ---

    Written & Maintained by: 
        Astromsoc
    Last Updated at:
        Apr 11, 2023
"""


import torch
import torch.nn as nn
from transformers import BertModel



class EasyReg(nn.Module):
    def __init__(self,
                 model_name: str='bert-base-uncased',
                 lstm_hidden_dim: int=256,
                 lstm_num_layers: int=2,
                 interim_linear_dim: int=1024,
                 dropout: float=0.3,
                 concat_wlen: bool=True,
                 concat_nsyl: bool=True):
        super().__init__()
        # archiving
        self.model_name         = model_name
        self.interim_linear_dim = interim_linear_dim
        self.dropout            = dropout
        self.lstm_hidden_dim    = lstm_hidden_dim
        self.lstm_num_layers    = lstm_num_layers
        self.concat_wlen        = concat_wlen
        self.concat_nsyl        = concat_nsyl
        # bert 
        self.bert_emb = BertModel.from_pretrained(model_name)
        # dropout
        self.dropout = dropout
        # rnn
        bidirectional = False
        self.roll = nn.LSTM(
            input_size=self.bert_emb.config.hidden_size,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_num_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        # flexibly add interim linear
        extra_ncount = int(concat_wlen) + int(concat_nsyl)
        self.reg_head = nn.Linear(extra_ncount + self.lstm_hidden_dim, 1) if interim_linear_dim <= 0 else nn.Sequential(
            nn.Linear(extra_ncount + self.lstm_hidden_dim, interim_linear_dim),
            nn.GELU(),
            nn.Linear(interim_linear_dim, 1)
        )
    

    def locked_dropout(self, x):
        """
            keep the same dropout for the entire batch
        """
        if (not self.training) or (not self.dropout):
            return x
        mask = x.new_empty(1, x.size(2)).bernoulli_(1 - self.dropout)
        mask = mask.div_(1 - self.dropout).expand_as(x)
        return x * mask

    
    def forward(self, ids, wlens, nsyls):
        # obtain embeddings
        x = self.locked_dropout(self.bert_emb(ids)[0])
        # go through LSTMs
        x = self.roll(x)[0][:, -1, :]
        # concat embeddings & additional attributes in one linear
        if self.concat_nsyl and self.concat_wlen: x = torch.cat((wlens, nsyls, x), dim=-1)
        elif self.concat_nsyl: x = torch.cat((nsyls, x), dim=-1)
        elif self.concat_wlen: x = torch.cat((wlens, x), dim=-1)
        # go through the regression head
        x = self.reg_head(x)
        return x



class DualReg(nn.Module):
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



class lockedDropoutLSTM(nn.Module):
    """
        Layered LSTM w/ locked dropouts.
    """
    def __init__(self, 
                 input_dim: int, 
                 lstm_dim: int=512, 
                 lstm_bi: bool=False,
                 lstm_layers: int=1, 
                 dropout: float=0.0):
        super().__init__()
        # archiving
        self.input_dim      = input_dim
        self.lstm_dim       = lstm_dim
        self.lstm_bi        = lstm_bi
        self.lstm_layers    = lstm_layers
        self.dropout        = dropout
        # build lstm layers
        self.lstms = nn.ModuleList([
            nn.LSTM(input_size=self.input_dim if i == 0 else self.lstm_dim * (1 + int(self.lstm_bi)),
                    hidden_size=self.lstm_dim,
                    num_layers=1,
                    bidirectional=self.lstm_bi,
                    batch_first=True)
        for i in range(self.lstm_layers)])

    
    def locked_dropout(self, x):
        """
            keep the same dropout for the entire batch
        """
        if (not self.training) or (not self.dropout): return x
        mask = x.new_empty(1, x.size(2)).bernoulli_(1 - self.dropout)
        mask = mask.div_(1 - self.dropout).expand_as(x)
        return x * mask
    

    def forward(self, x):
        for lstm in self.lstms:
            x = self.locked_dropout(lstm(x)[0])
        return x



class PTEmbReg(nn.Module):

    def __init__(self,
                 num_chr: int,
                 pad_idx: int,
                 emb_dim: int=128,
                 emb_dropout: float=0.3,
                 emb_linear_dims: list=[1024, 512, 256],
                 lstm_hidden_dim: int=256,
                 lstm_num_layers: int=2,
                 lstm_bidirectional: bool=True,
                 lstm_dropout: float=0.3,
                 use_lockedlstm: bool=True,
                 interim_linear_dim: int=32,
                 concat_wlen: bool=True,
                 concat_nsyl: bool=True):
        super().__init__()
        # archiving
        self.num_chr            = num_chr
        self.pad_idx            = pad_idx
        self.emb_dim            = emb_dim
        self.emb_dropout        = emb_dropout
        self.lstm_hidden_dim    = lstm_hidden_dim
        self.lstm_num_layers    = lstm_num_layers
        self.lstm_dropout       = lstm_dropout
        self.lstm_bidirectional = lstm_bidirectional
        self.use_lockedlstm     = use_lockedlstm
        self.last_ln            = interim_linear_dim
        self.concat_wlen        = concat_wlen
        self.concat_nsyl        = concat_nsyl
        extra_dim               = int(self.concat_wlen) + int(self.concat_nsyl)

        # build layers
        # >>>>> EMBEDDINGS <<<<<
        self.emb = nn.Embedding(num_embeddings=self.num_chr,
                                embedding_dim=self.emb_dim,
                                padding_idx=self.pad_idx)
        # >>>>>    LSTM    <<<<<
        self.run = nn.LSTM(
            input_size=self.emb_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_num_layers,
            bidirectional=self.lstm_bidirectional,
            dropout=self.lstm_dropout,
            batch_first=True
        ) if not self.use_lockedlstm else lockedDropoutLSTM(
            input_dim=self.emb_dim,
            lstm_dim=self.lstm_hidden_dim,
            lstm_bi=self.lstm_bidirectional,
            lstm_layers=self.lstm_num_layers,
            dropout=self.lstm_dropout
        )
        # >>>>>    CLS     <<<<<
        cls_indim = self.lstm_hidden_dim * (1 + int(self.lstm_bidirectional)) + extra_dim
        self.cls = nn.Linear(in_features=cls_indim, out_features=1) if self.last_ln > 0 else nn.Sequential(
                   nn.Linear(in_features=cls_indim, out_features=self.last_ln),
                   nn.ReLU(),
                   nn.Linear(in_features=self.last_ln, out_features=1))
    

    def locked_dropout(self, x):
        """
            keep the same dropout for the entire batch
            Args:
                x (batch_size, padded_seq_len, feature_dim)
        """
        if (not self.training) or (not self.emb_dropout): return x
        mask = x.new_empty(1, x.size(2)).bernoulli_(1 - self.emb_dropout)
        mask = mask.div_(1 - self.emb_dropout).expand_as(x)
        return x * mask


    def forward(self, chrs, wlens, nsyls):
        # embeddings (+ dropout) & linear layers
        x = self.locked_dropout(self.emb(chrs))
        # lstms
        x = self.run(x)
        # take out corresponding shape
        x = x[:, -1, :] if self.use_lockedlstm else x[0][:, -1, :]
        # whether to concat with extra attributes
        if self.concat_nsyl and self.concat_wlen:
            x = torch.cat((x, wlens, nsyls), dim=-1)
        elif self.concat_nsyl:
            x = torch.cat((x, nsyls), dim=-1)
        elif self.concat_wlen:
            x = torch.cat((x, wlens), dim=-1)
        # last classification layer(s)
        x = self.cls(x)
        return x




class ChrPlusWordEmbReg(nn.Module):

    def __init__(self,
                 num_chr: int,
                 pad_idx: int,
                 emb_dim: int=128,
                 dropout: float=0.3,
                 emb_linear_dims: list=[1024, 512, 256],
                 model_name: str='bert-base-uncased',
                 interim_linear_dim: int=32,
                 concat_wlen: bool=True,
                 concat_nsyl: bool=True):
        super().__init__()
        # archiving
        self.num_chr        = num_chr
        self.pad_idx        = pad_idx
        self.emb_dim        = emb_dim
        self.emb_dropout    = dropout
        self.emb_lnr_dims   = emb_linear_dims
        self.last_ln        = interim_linear_dim
        self.model_name     = model_name
        self.concat_wlen    = concat_wlen
        self.concat_nsyl    = concat_nsyl
        extra_dim           = int(self.concat_wlen) + int(self.concat_nsyl)

        # build layers
        self.bert_emb = BertModel.from_pretrained(self.model_name)
        self.emb = nn.Embedding(num_embeddings=self.num_chr,
                                embedding_dim=self.emb_dim,
                                padding_idx=self.pad_idx)
        self.emb_lns = nn.Sequential(*[l for i, in_dim in enumerate(self.emb_lnr_dims) for l in
            [nn.Linear(in_features=self.emb_dim if i == 0 else self.emb_lnr_dims[i - 1],
                       out_features=self.emb_lnr_dims[i]),
             nn.ReLU()]
        ])
        self.cls = nn.Linear(in_features=self.emb_lnr_dims[-1] + extra_dim, 
                             out_features=1) if self.last_ln > 0 else nn.Sequential(
                   nn.Linear(in_features=self.emb_lnr_dims[-1] + extra_dim, 
                             out_features=self.last_ln),
                   nn.ReLU(),
                   nn.Linear(in_features=self.last_ln,
                             out_features=1)
        )
    

    def locked_dropout(self, x):
        """
            keep the same dropout for the entire batch
        """
        if (not self.training) or (not self.emb_dropout):
            return x
        mask = x.new_empty(1, x.size(1)).bernoulli_(1 - self.emb_dropout)
        mask = mask.div_(1 - self.emb_dropout).expand_as(x)
        return x * mask


    def forward(self, chrs, wlens, nsyls):
        # characters go through embeddings (+ dropout) & linear layers
        x = self.emb(chrs).mean(dim=-2)
        x = self.locked_dropout(x)
        x = self.emb_lns(x)
        # whether to concat with extra attributes
        if self.concat_nsyl and self.concat_wlen:
            x = torch.cat((x, wlens, nsyls), dim=-1)
        elif self.concat_nsyl:
            x = torch.cat((x, nsyls), dim=-1)
        elif self.concat_wlen:
            x = torch.cat((x, wlens), dim=-1)
        # go through last classification layer(s)
        x = self.cls(x)
        return x





"""
    Mapping of model classes from config specification strings.
"""

ChooseYourModel = {
    'early-fused': EasyReg,
    'later-fused': DualReg,
    'pretrained-emb': PTEmbReg
}
