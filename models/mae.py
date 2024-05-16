'''Main module for the MAE.
'''


import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
from .patch_embed import PatchEmbed
from .transformer_encoder import TransformerEncoderNetwork



class MaskedAutoEncoder(nn.Module):
    '''Implementation of MAE with ViT.
    '''

    def __init__(self, 
                 patch_size, 
                 image_size, 
                 encoder_embedding_dim, 
                 decoder_embedding_dim, 
                 encoder_transformer_blocks_depth, 
                 decoder_transformer_blocks_depth, 
                 device, **kwargs):

        '''Init variables.
        '''


        #----------------------- Encoder -------------------------
        self.patch_embed = PatchEmbed(patch_size=patch_size, 
                                      image_depth=image_depth, 
                                      embedding_dim=encoder_embedding_dim, 
                                      init_std=0.02,
                                      device=device)

        self.num_patches = self.patch_embed.num_patches 
        self.init_std = init_std

        
        #we initialize with zeros because we will be populating them later with normal distribution or any other dist.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embedding_dim)) 
        #we will be using learnable parameters as positional embedding vector. The +1 is for the CLS token.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, encoder_embedding_dim)) 
        

        self.encoder_transformer_blocks = TransformerEncoderNetwork(device=device,
                                                            input_dim=encoder_embedding_dim,
                                                            transformer_network_depth=encoder_transformer_blocks_depth,
                                                            **kwargs).to(device)

        self.encoder_norm = nn.LayerNorm(encoder_embedding_dim).to(device)

        #-------------------------------------------------------

        #----------------------- Decoder ----------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(encoder_embedding_dim, decoder_embedding_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embedding_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embedding_dim))

        self.decoder_transformer_blocks = TransformerEncoderNetwork(device=device,
                                                                    input_dim=decoder_embedding_dim,
                                                                    transformer_network_depth=decoder_transformer_blocks_depth,
                                                                    **kwargs).to(device)
        
        self.decoder_norm = nn.LayerNorm(decoder_embedding_dim).to(device)

        self.decoder_output = nn.Linear(decoder_embedding_dim, self.num_patches**2 * image_depth, bias=True)
        
        self.apply(self.initialize_weights)


    def initialize_weights(self, m):
        '''Initialize the parameters.
        '''
        
        #initialize gaussian distribution on the weight parameters. Bias parameters will be 0 since some of them are initialized with 0 already.
        
        
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, std=self.std_init)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        torch.nn.init.trunc_normal_(self.cls_token.weights, std=self.init_std)
        torch.nn.init.trunc_normal_(self.encoder_pos_embed.weights, std=self.init_std)
        torch.nn.init.trunc_normal_(self.mask_token.weights, std=self.init_std)
        torch.nn.init.trunc_normal_(self.decoder_pos_embed.weights, std=self.init_std)
    

    def random_masking(self, x, mask_ratio):
        '''The strategy of random masking in MAE is:
            1) Randomly (normal dist) shuffle the embedded patches on patch level.
            2) Remove the last N patches based on the mask ratio.
            3) Use the remaining patches as input to the encoder.
            4) The output from the encoder will be then appended with mask tokens (the overall length of this output now will equal to the original length of the patch embedding vector).
            5) Then, this new vector will be unshuffled back to the original positions from before 1). The result will be mask tokens at places where the original elements were removed, and MAE's encoder's output at the non-removed elements.
            6) The decoder will take the vector from 5) as input and predict a vector of the original size input before 1).
            7) Finally, the output from decoder will be masked so that only the unmasked patches/pixels are compared for the loss.
        '''
    
    def forward_encoder(self, x, mask_ratio=None):
        '''Forward propagation for the encoder module.
        '''

        x = self.patch_embed(x) #patch embedding as in traditional ViT.

        x = x + self.encoder_pos_embed[:, 1:, :] #the cls token is not yet appended to the patch embedding. We will add the cls token after masking. Hence the pos embed is excluded of the cls token index as well.




        


        


        


