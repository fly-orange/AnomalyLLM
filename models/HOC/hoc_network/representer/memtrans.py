import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..Embed import DataEmbedding
from .RevIN import RevIN
from .attn import AttentionLayer

class EncoderLayer(nn.Module):
    def __init__(self, attn, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff if d_ff is not None else 4 * d_model
        self.attn_layer = attn
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x):
        '''
        x : N x L x C(=d_model)
        '''
        out = self.attn_layer(x)
        x = x + self.dropout(out)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)    # N x L x C(=d_model)
    

# Transformer Encoder
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        '''
        x : N x L x C(=d_model)
        '''
        for attn_layer in self.attn_layers:
            x = attn_layer(x)

        if self.norm is not None:
            x = self.norm(x)

        return x


class MemTrans(nn.Module):
    def __init__(self, configs):
        super(MemTrans, self).__init__()
        self.patch_size = configs.patch_size
        self.win_size = configs.window_size
        self.patch_num = self.win_size // self.patch_size
        self.embedding_patch_size = DataEmbedding(self.patch_size, configs.d_model)

         
        # Encoder
        dropout = 0.0
        activation = 'relu'
        
        self.mem = nn.Parameter(F.normalize(torch.rand((configs.input_channels, configs.num_memory, self.win_size), dtype=torch.float), dim=-1)) # (M, N, L)
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        self.win_size, configs.d_model, configs.n_heads, dropout=dropout
                    ), configs.d_model, configs.d_f, dropout=dropout, activation=activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer = nn.LayerNorm(configs.d_model)
        )
        # self.projection = nn.Linear(configs.d_model * configs.input_channels * self.patch_num, configs.project_channels, bias=True)
        self.projection= nn.Sequential(
            nn.Linear(configs.d_model * configs.input_channels * self.patch_num, configs.d_model * configs.input_channels * self.patch_num // 2),
            nn.BatchNorm1d(configs.d_model * configs.input_channels * self.patch_num // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.d_model * configs.input_channels * self.patch_num // 2, configs.project_channels),
        )


    def forward(self, x):
        B, M, L = x.shape # Batch win_size channel
        x = x.permute(0,2,1)

        ## Revin layer
        # revin_layer = RevIN(num_features=M)
        # x = revin_layer(x, 'norm') # (B, L, M)

        ## Instance norm
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev
        
        # mem_norm = (self.mem - means) / stdev
        mem = self.select(x, self.mem)  # (B, L, M)
        mem = (mem - means) / stdev

        x = rearrange(x, 'b l m -> b m l') #Batch channel win_size
        x = rearrange(x, 'b m (n p) -> (b m) n p', p = self.patch_size) 
        x = self.embedding_patch_size(x, None) # (b,m)  n, D

        mem = rearrange(mem, 'b l m -> b m l') #Batch channel win_size
        mem = rearrange(mem, 'b m (n p) -> (b m) n p', p = self.patch_size) 
        mem = self.embedding_patch_size(mem, None) # (b,m)  n, D


        x_all = torch.cat((mem, x), dim=1)   # (b,m) 2n, D
        x_all = self.encoder(x_all)[:, x_all.size(1)//2:]   # (b,m) n D
        x_f = rearrange(x_all, '(b m) n d -> b m n d', m = M).reshape(B,-1)
        x_f = self.projection(x_f)

        return x_f
    
    def select(self, x, mem):
        '''x: (B, L, M), mem (M, N, L)  ->  select_mem (B, L, M) '''
        # x_ori = self.embedding_window_size(x)
        x_tem = x.permute(2,0,1) # (M, B, L)
        attn_mat = torch.matmul(x_tem, mem.transpose(1,2)) 
        attn_mat /= (torch.norm(x_tem, p=2, dim=-1).unsqueeze(-1)*torch.norm(mem, p=2, dim=-1).unsqueeze(1)) # (M, B, N) cosine similarity
        _, attn_max = torch.max(attn_mat, dim=-1) # (M, B)
        x_s = torch.gather(mem, 1, attn_max.unsqueeze(-1).expand(-1,-1,x.size(1))) # (M, B, L)
        
        return x_s.permute(1,2,0) # (B, M, L)

    
