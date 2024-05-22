'''Patch Embeding layer.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
import torch.nn as nn
import einops.layers.torch as einops_torch


class PatchEmbed(nn.Module):
    '''The idea of patch embedding is to divide a given image into same-size grids and project each grids into another dimension linearly.
       This can be done effectively by using a conv layer with specific configuration. 
       Although, in this implementation, we will be doing it a bit manually using torch.unFold function. The reason being, during loss calculation,
       the original images has to be "patchified" again to perform the loss comparison. I believe it is better to keep this operation identical throughout the entire process.
       Having different ways of "patchifying" the images might contribute to information loss or inaccuracies. 
    '''


    #    def __init__(self, patch_size, image_depth, embedding_dim, device):
    #
    #        super(PatchEmbed, self).__init__()
    #
    #        self.patch_projection = nn.Conv2d(image_depth, embedding_dim, kernel_size=patch_size, stride=patch_size).to(device)
    #
    #    def forward(self, x):
    #
    #        x = self.patch_projection(x) #the output of this will be [batch size, embedding_dim, num_patches in height dimension, num_patches in width dimension]
    #        x = x.flatten(2) #flatten both the num_patches dimension to get the total number of patches. [batch size, embedding_dim, num_patches]
    #        x = x.transpose(1,2) #swap the axis. [batch size, num_patches, embedding_dim]
    #
    #        return x

    def __init__(self, patch_size, image_depth, embedding_dim, device):
        '''Init variables.
        '''

        super(PatchEmbed, self).__init__()
        
        self.unfolding_func = nn.Unfold(kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
        self.einops_rearrange = einops_torch.Rearrange('b e p -> b p e') #this is to change the position of the embedding and the number of patches dimension after Unfold.
        self.patch_linear_layer = nn.Linear(in_features=patch_size*patch_size*image_depth, out_features=embedding_dim, bias=True) #to linearly project the patches. 


    def get_non_overlapping_patches(self, imgs):
        '''imgs: image tensors of [B, C, H, W]
            Perform the unfolding operation and return the patch without the linear projection. This function can be used by the loss calculation module later.
        '''
        
        patched_image_tensors = self.unfolding_func(imgs)
        rearranged_tensors = self.einops_rearrange(patched_image_tensors) 
        self.num_patches = rearranged_tensors.shape[-2]

        return rearranged_tensors
    
    
    def forward(self, x):
        '''Creates linear projection out of the patches from the images.
        '''

        patched_image_tensors = self.get_non_overlapping_patches(x)
        linear_projected_patches = self.patch_linear_layer(patched_image_tensors)


        return linear_projected_patches



