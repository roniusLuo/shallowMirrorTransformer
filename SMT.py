"""
Shallow Mirror Transformer

Written by Jing Luo from Xi'an University of Technology, China.

luojing@xaut.edu.cn
"""

import numpy as np
import math
import random

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from einops import rearrange
from einops.layers.torch import Rearrange

seed_n = np.random.randint(500)

random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)

class PatchEmbedding(nn.Module):
    def __init__(self,  emb_size: int = 40, eeg_size: int = 1125, conv_len: int = 25):
        super().__init__()
        self.channel_num=3
        self.in_channels=1
        self.projection = nn.Sequential(          
            # Swap the time dimension of the EEG with the channel dimension
            Rearrange('b e h w -> b e w h'),
            nn.Conv2d(self.in_channels, emb_size, (conv_len, 1), stride=(1, 1)),
            nn.Conv2d(emb_size, emb_size, (1, self.channel_num), stride=(1, 1)),           
            nn.BatchNorm2d(emb_size, momentum=0.1, eps=1e-5),
            Rearrange('b e h w -> b (h w) e')
        )              
        self.pos_embed = nn.Parameter(torch.zeros(1, eeg_size-conv_len+1, emb_size))
        self.pos_drop = nn.Dropout(p=0.2)
       
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape 
        # feature extraction block
        x = self.projection(x)     
        # position embedding
        x = self.pos_drop(x + self.pos_embed)      
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 40, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        energy = energy / scaling
        att = F.softmax(energy, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.0):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 40,
                 drop_p: float = 0.0,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.0,
                 **kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 6, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, input_size: int = 1101, emb_size: int = 40, n_classes: int = 2):       
        super().__init__()
        self.poolk = 75
        self.pools = 15
        self.linear_len = (input_size-self.poolk)//self.pools+1
        self.pool = nn.AvgPool2d(kernel_size=(1,self.poolk),stride=(1,self.pools))
        self.drop = nn.Dropout(p = 0.5)
        self.logS = nn.LogSoftmax(dim=1)
        self.soft = nn.Softmax(dim=1)
        self.linear=nn.Linear(self.linear_len*emb_size,n_classes)
        
    def forward(self,x):       
        x = torch.square(x)
        # Exchange data in various dimensions
        x = x.permute(0,2,1)
        x = self.pool(x)
        x = torch.log(x)
        x = self.drop(x)
        x = self.logS(x)
        x = rearrange(x, "b h n -> b (n h)")
        x = self.linear(x)
        x=self.soft(x)
        return x

class SMT(nn.Sequential):
    def __init__(self,
                emb_size: int = 40,
                eeg_size: int = 1125,
                depth: int = 1,
                n_classes: int = 2,
                conv_len: int =25,
                **kwargs):
        super().__init__(
            PatchEmbedding(emb_size, eeg_size, conv_len),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(eeg_size-conv_len+1, emb_size, n_classes)
        )
