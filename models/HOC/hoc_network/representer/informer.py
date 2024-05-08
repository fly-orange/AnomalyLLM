import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils.masking import TriangularCausalMask, ProbMask
from .layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from .layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from .layers.Embed import DataEmbedding_wo_pos
import numpy as np


class Informer(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, configs):
        super(Informer, self).__init__()
        self.output_attention = False
        d_ff = 512
        activation = 'gelu'
        factor = 3 
        distil = True 

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.input_channels, configs.d_model)
        # self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
        #                                    configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=configs.dropout,
                                      output_attention=self.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    d_ff,
                    dropout=configs.dropout,
                    activation=activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        # self.decoder = Decoder(
        #     [
        #         DecoderLayer(
        #             AttentionLayer(
        #                 ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
        #                 configs.d_model, configs.n_heads),
        #             AttentionLayer(
        #                 ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
        #                 configs.d_model, configs.n_heads),
        #             configs.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             activation=configs.activation,
        #         )
        #         for l in range(configs.d_layers)
        #     ],
        #     norm_layer=torch.nn.LayerNorm(configs.d_model),
        #     projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        # )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = torch.mean(enc_out, dim=1)
        # dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return enc_out, attns
        else:
            return enc_out  # [B, L, D]