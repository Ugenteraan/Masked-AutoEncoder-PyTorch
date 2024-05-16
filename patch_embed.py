'''Patch Embeding layer.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
import torch.nn as nn



class PatchEmbed(nn.Module):
    '''The idea of patch embedding is to divide a given image into same-size grids and project each grids into another dimension linearly.
       This can be done effectively by using a conv layer with specific configuration.
    '''


    def __init__(self, patch_size, image_depth, embedding_dim, device):

        super(PatchEmbed, self).__init__()

        self.patch_projection = nn.Conv2d(image_depth, embedding_dim, kernel_size=patch_size, stride=patch_size).to(device)

    def forward(self, x):

        x = self.patch_projection(x) #the output of this will be [batch size, embedding_dim, num_patches in height dimension, num_patches in width dimension]
        x = x.flatten(2) #flatten both the num_patches dimension to get the total number of patches. [batch size, embedding_dim, num_patches]
        x = x.transpose(1,2) #swap the axis. [batch size, num_patches, embedding_dim]

        return x
