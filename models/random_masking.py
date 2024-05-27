'''Random masking strategy for MAE.
Important concept to note.
            Say we have a tensor x. When torch.argsort is called on the tensor, the result (let's call it y) is a tensor of indices that denote the location (index) of the elements from smallest to the biggest. We could perform torch.gather function on tensor x using tensor y as incidices to get tensor x_ that are arranged in ascending order of value. Now, if we want to reverse back the arrangement of x_ to x, we need to call another torch.argsort on y. This will give us the indices in ascending order of the indices in y. Let's call this z. Now when z is applied on x_, we get x. 
            Example:
                x = [0.7, 0.9, 0.1, 0.2, 0.8] #original array.
                y = [2, 3, 0, 4, 1] #indices that denote the elements of x from smallest to biggest.
                z = [2, 4, 0, 1, 3] #indices that denote the elements of y from smallest to biggest.
                x_ = [0.1, 0.2, 0.7, 0.8, 0.9] #y applied on x.
                x = [0.7, 0.9, 0.1, 0.2, 0.8] #z applied on x_.
        '''

import torch

class RandomMasking:
    '''Implemented separately to include explanations that provide more clarity on the codes.
    '''
    
    def __init__(self, masking_ratio, device):
        '''Variable init.
        '''

        self.masking_ratio = masking_ratio
        self.device = device


    def __call__(self, x):
        '''Remember, in this case we need to perform random shuffling on the tensor and the shuffle should be done per batch (i.e. different samples, different shuffle order).
        '''
        
        batch_num, patch_num, embedding_dim = x.shape # batch number, num of patches, dimension of each patch.
        
        len_keep = int(patch_num * (1-self.masking_ratio)) #this will give the number of patches to retain.

        random_noise = torch.randn(batch_num, patch_num, device=self.device) #note that the noise is generated per batch. This is to ensure each batch gets a unique shuffling order.

        idxs_random_shuffle = torch.argsort(random_noise, dim=1)
        idxs_reverse_shuffle = torch.argsort(idxs_random_shuffle, dim=1)

        idxs_to_keep = idxs_random_shuffle[:, :len_keep] #the subset of patches to keep. 
        x_masked = torch.gather(x, dim=1, index=idxs_to_keep.unsqueeze(-1).repeat(1, 1, embedding_dim)) #we need to repeat the idx_to_keep in the last dimension to match tensor x.

        #during the loss calculation, only the masked patches will be used. To do that, we need a mask. It'll be way easier if the mask is generated here itself.
        inverted_masks = torch.ones([batch_num, patch_num], device=self.device) #1 means keep, 0 means mask away.
        inverted_masks[:, :len_keep] = 0 #mask according to ratio first. REMEMBER! The mask will have more 1s than 0s cause this is an inverted mask for the loss calculation. Not to actually mask the images.
        inverted_masks = torch.gather(inverted_masks, dim=1, index=idxs_reverse_shuffle).to(device) #invert the shuffle so we get the masks at the appropriate locations.
        
        return x_masked, inverted_masks, idxs_reverse_shuffle

