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


    def initialize_weights(self):
        '''Initialize the parameters.
        '''
        
        #initialize gaussian distribution on the weight parameters. Bias parameters will be 0 since some of them are initialized with 0 already.
        torch.nn.init.trunc_normal_(self.cls_token.weights, std=self.init_std)
        torch.nn.init.trunc_normal_(self.encoder_pos_embed.weights, std=self.init_std)
        torch.nn.init.trunc_normal_(self.mask_token.weights, std=self.init_std)
        torch.nn.init.trunc_normal_(self.decoder_pos_embed.weights, std=self.init_std)
        
        torch.nn.init.xavier_uniform_(self.patch_projection.weights) 







        


        


        


