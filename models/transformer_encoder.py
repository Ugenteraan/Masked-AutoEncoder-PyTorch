'''Transformer blocks that contain multi-head attention modules and feed forward layers. 
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from .mhsa import MultiHeadAttention
from .feedforward_block import FeedForwardBlock

class TransformerBlock(nn.Module):
    '''A single transformer block.
    '''


    def __init__(self, input_dim, num_heads, attn_dropout_prob, mlp_ratio, feedforward_dropout_prob, device):

        super(TransformerBlock, self).__init__()

        #initialize the self attention module together with a layernorm layer.
        self.multi_head_attention_block = nn.Sequential(
                                                        nn.LayerNorm(input_dim),
                                                        MultiHeadAttention(input_dim=input_dim,
                                                                           num_heads=num_heads,
                                                                           attn_dropout_prob=attn_dropout_prob,
                                                                           device=device).to(device)
                                                        )

        #initialize the feedforward block together with a layernorm layer.
        self.feedforward_block = nn.Sequential(
                                                nn.LayerNorm(input_dim),
                                                FeedForwardBlock(input_dim=input_dim,
                                                                 mlp_ratio=mlp_ratio,
                                                                 feedforward_dropout_prob=feedforward_dropout_prob, 
                                                                 device=device).to(device)
                                                )
        

    def forward(self, x):

        multihead_attention_output = self.multi_head_attention_block(x)
        multihead_attention_output += x #skip connection 

        feedforward_output = self.feedforward_block(multihead_attention_output)
        feedforward_output += multihead_attention_output

        return feedforward_output




class TransformerNetwork(nn.Sequential):
    '''Created a network of transformers using the TransformerBlock module with the given depth.
    '''
    
    def __init__(self, transformer_network_depth, input_dim, device, **kwargs):

        super().__init__(*[TransformerBlock(input_dim=input_dim,
                                                   device=device,
                                                   **kwargs
                                                   ).to(device) for _ in range(transformer_network_depth)])



