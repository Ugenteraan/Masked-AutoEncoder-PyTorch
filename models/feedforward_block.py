'''Implementation of a single feedforward block in a transformer.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

class FeedForwardBlock(nn.Sequential):

    def __init__(self, input_dim, mlp_ratio, feedforward_dropout_prob, device):

        #let's define the sequence using the nn.sequential's init itself.
        super().__init__(
                nn.Linear(input_dim, input_dim*mlp_ratio).to(device),
                nn.GELU(),
                nn.Dropout(feedforward_dropout_prob).to(device),
                nn.Linear(input_dim*mlp_ratio, input_dim).to(device)
                )


