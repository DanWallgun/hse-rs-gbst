import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from charformer_pytorch import GBST


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class EmbeddingPosEncoder(nn.Module):
    def __init__(self, ntoken, d_model, dropout):
        super().__init__()
        self.d_model = d_model
        self.encoder = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
    def forward(self, x):
        x = self.encoder(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, conv_ks: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.token_encoder = EmbeddingPosEncoder(ntoken, d_model, dropout)
        
        self.conv = nn.Conv1d(d_model, d_model, conv_ks)
        self.transformer_encoder = TransformerEncoder(TransformerEncoderLayer(d_model, nhead, d_hid, dropout), nlayers)

        self.decoder = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.token_encoder(x)
        
        x = self.conv(x.transpose(2, 1)).transpose(2, 1)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)

        x = self.decoder(x)
        return x


class GBSTEncoder(nn.Module):
    def __init__(self, ntoken, d_model):
        super().__init__()
        self.gbst = GBST(
            num_tokens=ntoken,          # number of tokens, should be 256 for byte encoding (+ 1 special token for padding in this example)
            dim=d_model,                # dimension of token and intra-block positional embedding
            max_block_size=4,           # maximum block size
            downsample_factor=4,        # the final downsample factor by which the sequence length will decrease by
            score_consensus_attn=True   # whether to do the cheap score consensus (aka attention) as in eq. 5 in the paper
        )
    def forward(self, x):
        x, _, scores = self.gbst(x)
        return x


class GBSTTransformerModel(TransformerModel):
    def __init__(self, ntoken: int, d_model: int, *args, **kwargs):
        super().__init__(ntoken, d_model, *args, **kwargs)
        self.token_encoder = GBSTEncoder(ntoken, d_model)
