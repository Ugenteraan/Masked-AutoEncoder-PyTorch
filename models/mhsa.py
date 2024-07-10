'''Multi-Head Self Attention mechanism module.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
# import einops.layers.torch as einops
import einops


class MultiHeadAttention(nn.Module):
    '''Einops implementation of multi-head self attention.
    '''

    def __init__(self, input_dim, num_heads, attn_dropout_prob, device):

        super(MultiHeadAttention, self).__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim//num_heads
        self.projection_keys_dim = self.head_dim
        self.projection_values_dim = self.head_dim
        self.attn_dropout_prob = attn_dropout_prob
        self.device = device

        
        #we can simplify the operation by multiplying the projection dimension by 2 and separate the query and keys since they are both projected to the same dimension.
        #if the values were projected to the same dimension we could have simply multiplied by 3 and performed the same operation. But we want to allow flexibility by for the value dimension (projected).
        self.Wq_Wk = nn.Linear(self.input_dim, self.projection_keys_dim*2).to(device) #weights to project the last dimension of the input tensor to a projecction dimemsion for the query and keys.
        self.Wv = nn.Linear(self.input_dim, self.projection_values_dim).to(device) #the weight to project the last dimension of the input tensor to a projection dimensions for the values.

        self.attention_dropout = nn.Dropout(self.attn_dropout_prob).to(device)

        #to project the values' dimension back to the patch embedding dimension.
        self.W_o = nn.Linear(self.projection_values_dim, self.input_dim).to(device)


        # self.einops_rearrange_qk = einops.Rearrange('b n (h e qk) -> (qk) b h n e', h=self.num_heads, qk=2).to(device) #Einops operation to rearrange the q & k and to create the head dimension.
        # self.einops_rearrange_v = einops.Rearrange('b n (h e k) -> k b h n e', h=self.num_heads, k=1).to(device)

        # self.einops_mhsa_concat = einops.Rearrange('b h n e -> b n (h e)').to(device) #remember, we want to concatenate the heads together at the end.


    def forward(self, x):
        
        qk = self.Wq_Wk(x) #get the keys and queries from the input.
        v = self.Wv(x) #get the values from the input.

        #apply the einops operations.
        # qk_rearranged = self.einops_rearrange_qk(qk)
        # v_rearranged = self.einops_rearrange_v(v)

        #apply einops from a diff library
        qk_rearranged = einops.rearrange(qk, 'b n (h e qk) -> (qk) b h n e', h=self.num_heads, qk=2)
        v_rearranged = einops.rearrange(v, 'b n (h e k) -> k b h n e', h=self.num_heads, k=1)

        queries, keys = qk_rearranged[0], qk_rearranged[1]
        values = v_rearranged[0]

        #perform the dot projection between the queries and the keys.
        dot_projection = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        
        #square root of the projected dimension of keys/queries.
        scaling_factor = self.projection_keys_dim ** 0.5

        scaled_dot_projection = F.softmax(dot_projection, dim=-1)/scaling_factor #softmax the last dimension and scale it.
        scaled_dot_projection = self.attention_dropout(scaled_dot_projection) #apply dropout if any.

        #calculate the attention 
        attention_qkv = torch.einsum('bhsl, bhlv -> bhsv', scaled_dot_projection, values)

        #concat all the heads
        # multi_head_concatenated = self.einops_mhsa_concat(attention_qkv)

        #apply einops from a diff library
        multi_head_concatenated = einops.rearrange(attention_qkv, 'b h n e -> b n (h e)')
        
        #project back to the original dimension.
        multi_head_projection = self.W_o(multi_head_concatenated)

        return multi_head_projection


