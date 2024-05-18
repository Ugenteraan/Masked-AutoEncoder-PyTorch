'''To patchify the images for the loss calculation.
'''


import torch
import torch.nn as nn
import einops.layers.torch as einops_torch


class Patchify:

    def __init__(self, patch_size):
        '''Variables init.
        '''

        self.patch_size = patch_size

        #initialize the unfold function.
        self.unfolding_func = nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size))
        self.einops_ops = einops_torch.Rearrange('b e p -> b p e') #change the position of the embedding (flattened image patch) and the num of patch


    def __call__(self, imgs):
        '''Patchify the images.
        '''
        patched_image_tensors = self.unfolding_func(imgs)

        rearranged_image_tensors = self.einops_ops(patched_image_tensors)

        return rearranged_image_tensors
    

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x




if __name__ == '__main__':

    imgs = torch.randn(2, 3, 224, 224)

    p = Patchify(patch_size=14)

    x = p(imgs)
    print(x.size())

    



