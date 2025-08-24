from typing import Type

import torch
import torch.nn as nn

class CasualAttention(nn.Module):
    def __init__(
            self, 
            d_in:int, 
            d_out:int, 
            context_length:int,
            dropout:float,
            qkv_bias:bool=False
            ):
        super().__init__()
        self.d_out:int = d_out
        self.W_query:Type[nn.Linear] = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key:Type[nn.Linear] = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value:Type[nn.Linear] = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(
                torch.ones(
                    context_length,
                    context_length
                ),
                diagonal=1
            )
        )

    def forward(self, x:Type[torch.Tensor]):
        b, num_tokens, d_in = x.shape
        
        queries:Type[torch.Tensor] = self.W_query(x)
        keys:Type[torch.Tensor] = self.W_key(x)
        values:Type[torch.Tensor] = self.W_value(x)

        attn_scores:Type[torch.Tensor] = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[
                :num_tokens, 
                :num_tokens
                ],
            -torch.inf
        )
        attn_weights:Type[torch.Tensor] = torch.softmax(
            attn_scores / keys.shape[-1]**0.5,
            dim=-1
        )
        context_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec

if __name__ == '__main__':
    d_in:int = 3
    d_out:int = 2

    inputs:Type[torch.Tensor] = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts   (x^3)
    [0.22, 0.58, 0.33], # with     (x^4)
    [0.77, 0.25, 0.10], # one      (x^5)
    [0.05, 0.80, 0.55]] # step     (x^6)
    )

    torch.manual_seed(123)

    batch = torch.stack((inputs, inputs), dim=0)
    context_lenght:int = batch.shape[1]

    ca = CasualAttention(
        d_in,
        d_out,
        context_lenght,
        0.0
        )
    
    context_vecs = ca(batch)

    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)