'''Main module for the MAE.
'''


import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from .patch_embed import PatchEmbed
from .transformer_encoder import TransformerNetwork 
from .random_masking import RandomMasking



class MaskedAutoEncoder(nn.Module):
    '''Implementation of MAE with ViT.
        The strategy of random masking in MAE is:
            1) Randomly (normal dist) shuffle the embedded patches on patch level.
            2) Remove the last N patches based on the mask ratio.
            3) Use the remaining patches as input to the encoder.
            4) The output from the encoder will be then appended with mask tokens (the overall length of this output now will equal to the original length of the patch embedding vector).
            5) Then, this new vector will be unshuffled back to the original positions from before 1). The result will be mask tokens at places where the original elements were removed, and MAE's encoder's output at the non-removed elements.
            6) The decoder will take the vector from 5) as input and predict a vector of the original size input before 1).
            7) Finally, the output from decoder will be masked so that only the unmasked patches/pixels are compared for the loss.
    '''

    def __init__(self, 
                 patch_size, 
                 image_size, 
                 image_depth,
                 encoder_embedding_dim, 
                 decoder_embedding_dim, 
                 encoder_transformer_blocks_depth, 
                 decoder_transformer_blocks_depth, 
                 masking_ratio,
                 normalize_pixel,
                 device,
                 encoder_num_heads,
                 decoder_num_heads,
                 encoder_mlp_ratio,
                 decoder_mlp_ratio,
                 attn_dropout_prob,
                 feedforward_dropout_prob,
                 logger=None,
                 init_std=0.02):

        '''Init variables.
        '''

        super(MaskedAutoEncoder, self).__init__()

        self.normalize_pixel = normalize_pixel
        self.random_masking = RandomMasking(masking_ratio=masking_ratio,
                                         device=device)

        #----------------------- Encoder -------------------------
        self.patch_embed = PatchEmbed(patch_size=patch_size, 
                                      image_size=image_size,
                                      image_depth=image_depth, 
                                      embedding_dim=encoder_embedding_dim, 
                                      device=device)

        self.num_patches = (image_size//patch_size)**2
        self.init_std = init_std

        
        #we initialize with zeros because we will be populating them later with normal distribution or any other dist.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embedding_dim)).to(device) 
        #we will be using learnable parameters as positional embedding vector. The +1 is for the CLS token.
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, encoder_embedding_dim)).to(device)
        

        self.encoder_transformer_blocks = TransformerNetwork(device=device,
                                                            input_dim=encoder_embedding_dim,
                                                            transformer_network_depth=encoder_transformer_blocks_depth,
                                                            num_heads=encoder_num_heads,
                                                            mlp_ratio=encoder_mlp_ratio,
                                                            attn_dropout_prob=attn_dropout_prob,
                                                            feedforward_dropout_prob=feedforward_dropout_prob).to(device)

        self.encoder_norm = nn.LayerNorm(encoder_embedding_dim).to(device)

        #-------------------------------------------------------

        #----------------------- Decoder ----------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(encoder_embedding_dim, decoder_embedding_dim, bias=True).to(device)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embedding_dim)).to(device)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embedding_dim)).to(device) #+1 for the cls token at dim 1.

        self.decoder_transformer_blocks = TransformerNetwork(device=device,
                                                                    input_dim=decoder_embedding_dim,
                                                                    transformer_network_depth=decoder_transformer_blocks_depth,
                                                                    num_heads=decoder_num_heads,
                                                                    mlp_ratio=encoder_mlp_ratio,
                                                                    attn_dropout_prob=attn_dropout_prob,
                                                                    feedforward_dropout_prob=feedforward_dropout_prob).to(device)
        
        self.decoder_norm = nn.LayerNorm(decoder_embedding_dim).to(device)

        self.decoder_output = nn.Linear(decoder_embedding_dim, patch_size**2 * image_depth, bias=True).to(device)

        self.mse_loss = nn.MSELoss(reduction='mean').to(device)
        
        self.apply(self.initialize_weights)


    def initialize_weights(self, m):
        '''Initialize the parameters.
        '''
        
        #initialize gaussian distribution on the weight parameters. Bias parameters will be 0 since some of them are initialized with 0 already.
        
        
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        #torch.nn.init.trunc_normal_(self.cls_token.weights, std=self.init_std)
        #torch.nn.init.trunc_normal_(self.encoder_pos_embed.weights, std=self.init_std)
        #torch.nn.init.trunc_normal_(self.mask_token.weights, std=self.init_std)
        #torch.nn.init.trunc_normal_(self.decoder_pos_embed.weights, std=self.init_std)
    

    
    def forward_encoder(self, x):
        '''Forward propagation for the encoder module.
        '''

        x = self.patch_embed(x) #patch embedding as in traditional ViT.
        
        x = x + self.encoder_pos_embed[:, 1:, :] #the cls token is not yet appended to the patch embedding. We will add the cls token after masking. Hence the pos embed is excluded of the cls token index as well.

        x, mask, idxs_reverse_shuffle  = self.random_masking(x) #perform random masking per batch.

        #append cls token
        cls_token = self.cls_token + self.encoder_pos_embed[:, :1, :] #add the positional embedding for cls token only.
        cls_token = cls_token.expand(x.shape[0], -1, -1) #-1 means not changing the size of that dimension.
        x = torch.cat((cls_token, x), dim=1)

        x = self.encoder_transformer_blocks(x)

        x = self.encoder_norm(x)

        return x, mask, idxs_reverse_shuffle


    
    def forward_decoder(self, x, idxs_reverse_shuffle):
        '''Forward propagation of the decoder module.
        '''

        x = self.decoder_embed(x) #linear projection

        #mask tokens need to be appended at the masked positions. This is the reason why we need the indices to reverse the shuffle.
        mask_tokens = self.mask_token.repeat(x.shape[0], idxs_reverse_shuffle.shape[1] + 1 - x.shape[1], 1) #in the 2nd element, we add 1 since x has cls token in it. So it's done to counteract that.
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1) #without CLS token.
        x_ = torch.gather(x_, dim=1, index=idxs_reverse_shuffle.unsqueeze(-1).repeat(1, 1, x.shape[2])) #this will unshuffle the tensor to the original position. In other words, the masked patches will now have mask tokens in their place while the unmasked patches will have the output from the encoder.
        x = torch.cat([x[:, :1, :], x_], dim=1) #append back the CLS token from the original x.

        x = x + self.decoder_pos_embed 

        x = self.decoder_transformer_blocks(x)

        x = self.decoder_norm(x)
        
        x = self.decoder_output(x)
        
        x = x[:, 1:, :] #remove cls token for the pretraining.

        return x
    

    def loss_calc(self, imgs, preds, masks):
        '''Calculates the loss of MAE.
        '''

        targets = self.patch_embed.get_non_overlapping_patches(imgs) #patched tensors without the linear projection.
        
        if self.normalize_pixel:
            mean = targets.mean(dim=-1, keepdim=True)
            var = targets.var(dim=-1, keepdim=True)
            targets = (targets-mean)/(var + 1.e-6)**.5 #the epsilon number was referenced from the official MAE implementation.
        
        loss = (preds - targets)**2
        loss = loss.mean(dim=-1)
        
        #Remember, we only want the loss to be calculated at the masked areas. Hence the mask multiplication.
        loss = (loss * masks).sum() / masks.sum()
        return loss


    
    def forward(self, x):

        latent, inverted_masks, idxs_reverse_shuffle = self.forward_encoder(x)

        preds = self.forward_decoder(x=latent, idxs_reverse_shuffle=idxs_reverse_shuffle)

        loss = self.loss_calc(imgs=x, preds=preds, masks=inverted_masks)
        
        return loss, preds, inverted_masks




