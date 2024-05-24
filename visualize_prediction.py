'''Module to visualize the prediction done by the decoder.
'''

import matplotlib.pyplot as plt
import torch
import numpy as np

from models.patch_embed import PatchEmbed


class VisualizePrediction:

    def __init__(self, 
                 visualize_batch_size=6, 
                 image_size=224,
                 image_depth=3,
                 patch_size=16,
                 fig_savepath='./figures/',
                 num_figs=5):


        self.image_size = image_size 
        self.visualize_batch_size = visualize_batch_size
        self.patch_size = patch_size
        self.fig_savepath = fig_savepath
        self.num_figs = num_figs
        self.patch_embed = PatchEmbed(patch_size=patch_size, 
                                 image_size=image_size,
                                 image_depth=image_depth)
        
    
    def plot_images(self, fig, axes, row_idx, original_image, masked_image, pred_image):
        '''Plot images using matplotlib.
        '''
        
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        
        #plot the original image 
        plt.sca(axes[row_idx, 0])
        original_image = np.transpose(original_image[0].cpu().numpy(), (1, 2, 0))
        original_image = (original_image * np.asarray(std)) + np.asarray(mean)
        plt.imshow(original_image)
        plt.axis('off')
        
        
        plt.sca(axes[row_idx, 1])
        masked_image = np.transpose(masked_image[0].cpu().numpy(), (1, 2, 0))
        masked_image = (masked_image * np.asarray(std)) + np.asarray(mean)
        plt.imshow(masked_image)
        plt.axis('off')

        plt.sca(axes[row_idx, 2])
        pred_image = np.transpose(pred_image[0].cpu().numpy(), (1, 2, 0))
        pred_image = (pred_image * np.asarray(std)) + np.asarray(mean)
        plt.imshow(pred_image)
        plt.axis('off')


    def mask_target(self, target, inverted_mask):
        '''In order to get the masked/unmasked area, the target image has to be patchified, applied with the mask, and unpatchified again.
        '''
        patchified = self.patch_embed.get_non_overlapping_patches(target)
        
        mask = 1 - inverted_mask.type(torch.int64) #invert the inverted mask
        
        mask = mask.unsqueeze(0).unsqueeze(-1).expand(-1, -1, patchified.size(-1))
        
        
        masked_patches = patchified*mask
        
        images = self.patch_embed.make_patches_into_images(patches=masked_patches)
        return images

        
    def plot(self,
             pred_tensor,
             target_tensor,
             inverted_masks,
             epoch_idx):

        '''Plots both the target and the prediction from the decoder side by side.
        '''

        fig, axes = plt.subplots(nrows=self.visualize_batch_size, ncols=3)
        
        
        for idx in range(self.visualize_batch_size):
            
            prediction = pred_tensor[idx]
            target = target_tensor[idx].unsqueeze(0)
            inverted_mask = inverted_masks[idx]
            
            #the unsqueeze is needed since Fold function requires a 4D tensor.
            prediction_image = self.patch_embed.make_patches_into_images(patches=prediction.unsqueeze(0))
            
            masked_image = self.mask_target(target=target, inverted_mask=inverted_mask)
            
            # print("SIZES:", target.size(), masked_image.size(), prediction_image.size())
            
            self.plot_images(fig=fig, 
                             axes=axes, 
                             row_idx=idx, 
                             original_image=target, 
                             masked_image=masked_image, 
                             pred_image=prediction_image)
            
        plt.savefig(f'{self.fig_savepath}/visualization - {epoch_idx}.jpg')
