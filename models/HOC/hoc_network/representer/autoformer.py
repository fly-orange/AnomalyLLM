import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from .layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,y
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Autoformer, self).__init__()

        self.output_attention = False
        factor = 3
        moving_avg = 25
        activation = 'gelu'
        d_ff = 512
        # Decomp
        kernel_size = moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.input_channels, configs.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=configs.dropout,
                                        output_attention=self.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=configs.dropout,
                    activation=activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
      

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        # mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        # zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        # seasonal_init, trend_init = self.decomp(x_enc)
        # # decoder input
        # trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        # seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # print(enc_out.shape)
        # dec
        # dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        # seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
        #                                          trend=trend_init)
        # # final
        # dec_out = trend_part + seasonal_part

        if self.output_attention:
            return enc_out, attns
        else:
            return torch.mean(enc_out, dim=1)  # [B, L, D]