import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from .layers.SelfAttention_Family import ReformerLayer
from .layers.Embed import DataEmbedding_wo_pos
import numpy as np


class Reformer(nn.Module):
    """
    Reformer with O(LlogL) complexity
    - It is notable that Reformer is not proposed for time series forecasting, in that it cannot accomplish the cross attention.
    - Here is only one adaption in BERT-style, other possible implementations can also be acceptable.
    - The hyper-parameters, such as bucket_size and n_hashes, need to be further tuned.
    The official repo of Reformer (https://github.com/lucidrains/reformer-pytorch) can be very helpful, if you have any questiones.
    """

    def __init__(self, configs):
        super(Reformer, self).__init__()

        self.output_attention = False
        d_ff = 512
        activation = 'gelu'
        bucket_size = 4
        n_hashes = 4

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.input_channels, configs.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(None, configs.d_model, configs.n_heads, bucket_size=bucket_size,
                                  n_hashes=n_hashes),
                    configs.d_model,
                    d_ff,
                    dropout=configs.dropout,
                    activation=activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # add placeholder
        # x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        # x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)
        # Reformer: encoder only
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # print(enc_out.shape)
        # enc_out = self.projection(enc_out)
        enc_out = torch.mean(enc_out, dim=1)

        if self.output_attention:
            return enc_out, attns
        else:
            return enc_out  # [B, L, D]