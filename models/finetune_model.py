'''Model to initialize for the fine-tune task.
'''
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import einops

from .patch_embed import PatchEmbed
from .transformer_encoder import TransformerNetwork 
from .positional_encoder import PositionalEncoder


class PretrainedEncoder(nn.Module):
    '''This module has the same configuration as the MaskedAutoEncoder's encoder without the masking part. Will be used to load the pretrained weights here for the fine-tuning.
    '''


    def __init__(self, 
                 patch_size, 
                 image_size, 
                 image_depth,
                 encoder_embedding_dim, 
                 encoder_transformer_blocks_depth, 
                 device,
                 encoder_num_heads,
                 encoder_mlp_ratio,
                 attn_dropout_prob,
                 feedforward_dropout_prob,
                 init_std=0.02,
                 logger=None,
                 using_tensorboard=False): #we need this parameter to disable gradients in some tensors due to tensorboard not being able to convert them into constants.

    
    
        super(PretrainedEncoder, self).__init__()

        
        #----------------------- Encoder -------------------------
        self.patch_embed = PatchEmbed(patch_size=patch_size, 
                                      image_size=image_size,
                                      image_depth=image_depth, 
                                      embedding_dim=encoder_embedding_dim, 
                                      device=device).to(device)

        self.num_patches = (image_size//patch_size)**2
        self.init_std = init_std

        
        #we initialize with zeros because we will be populating them later with normal distribution or any other dist.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embedding_dim), requires_grad=not using_tensorboard).to(device) 

        self.encoder_pos_embed = PositionalEncoder(token_length=self.num_patches+1, 
                                                   output_dim=encoder_embedding_dim, 
                                                   n=10000, 
                                                   device=device)

        self.encoder_transformer_blocks = TransformerNetwork(device=device,
                                                            input_dim=encoder_embedding_dim,
                                                            transformer_network_depth=encoder_transformer_blocks_depth,
                                                            num_heads=encoder_num_heads,
                                                            mlp_ratio=encoder_mlp_ratio,
                                                            attn_dropout_prob=attn_dropout_prob,
                                                            feedforward_dropout_prob=feedforward_dropout_prob).to(device)

        self.encoder_norm = nn.LayerNorm(encoder_embedding_dim).to(device)

        #-----------------------------------------------------------

    def forward(self, x):

        x = self.patch_embed(x)

        encoder_pos_embed_tensor = self.encoder_pos_embed()
        encoder_pos_embed_tensor = einops.repeat(encoder_pos_embed_tensor.unsqueeze(0), '() p e -> b p e', b=x.size(0))

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = x + encoder_pos_embed_tensor

        x = self.encoder_transformer_blocks(x)
        x = self.encoder_norm(x)

        return x

    
class FineTuneModelClassification(nn.Module):
    '''This module is the extra added layers used for the downstream tasks of classifications.
    '''

    def __init__(self,
                 input_dim,
                 expansion_factor,
                 num_class,
                 device, 
                 logger,
                 use_tensorboard=False):
        
        super(FineTuneModelClassification, self).__init__()

        self.device = device

        #we will be using the CLS token to perform the fine-tuning and classification tasks. 
        self.classification_head = nn.Sequential(nn.LayerNorm(input_dim),
                                                 nn.Linear(input_dim, input_dim*expansion_factor),
                                                 nn.GELU(),
                                                 nn.Linear(input_dim*expansion_factor, num_class))


    def forward(self, x):

        extracted_cls_token = x[:, 0, :].squeeze(1) #we don't need the 2nd dimension anymore.

        x = self.classification_head(extracted_cls_token)

        return x



    


