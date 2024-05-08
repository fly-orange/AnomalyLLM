from torch import nn
import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from einops import rearrange
from .Embed import DataEmbedding, DataEmbedding_wo_pos
from .representer import TCN, TimesNet, MemTrans, Autoformer, Reformer, Informer


representer_family={
     "tcn": TCN,
     "timesnet": TimesNet,
     "memtrans": MemTrans
}

class base_Model(nn.Module):
    def __init__(self, configs, device):
        super(base_Model, self).__init__()
        # simple block
        self.representer = representer_family[configs.reclass](configs).to(device)
        # FPT block
        self.center_predictor = FPT(configs, device).to(device)
        # self.center_predictor = Autoformer(configs).to(device)
  

    def forward(self, x_in): 
        '''x_in: (B, d, T)'''
        if torch.isnan(x_in).any():
            print('tensor contain nan')
        project = self.representer(x_in)

        '''FPT'''
        x_enc = x_in.permute(0, 2, 1)
        # x_enc = x_in
        center = self.center_predictor(x_enc) # (B, T, D)
        

        return project, center



class FPT(nn.Module):
    def __init__(self, configs, device):
        super(FPT, self).__init__()
        self.patch_size = 16
        self.d_model = 768
        self.stride = configs.stride
        self.d_ff = configs.d_ff
        
        self.padding_patch_layer = nn.ReplicationPad1d((0, configs.stride)) 
        self.enc_embedding = DataEmbedding_wo_pos(configs.input_channels * self.patch_size, self.d_model)

        self.gpt2 = GPT2Model.from_pretrained('models/HOC/hoc_network/gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name: # ln = layer normalization; wpe = position embedding
                param.requires_grad = True
            elif 'mlp' in name and configs.mlp == 1:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if configs.use_gpu:
            # device = torch.device('cuda:{}'.format(0))
            self.gpt2.to(device=device)


        patch_num = (configs.window_size  - self.patch_size) // configs.stride + 2
        self.out_layer = nn.Linear(configs.d_ff, configs.project_channels, bias=True)


    def forward(self, x_enc):
        B, L, M = x_enc.shape
        # print(x_enc.shape)
        # Normalization from Non-stationary Transformer

        # seg_num = 16
        # x_enc = rearrange(x_enc, 'b (n s) m -> b n s m', s=seg_num)
        # means = x_enc.mean(2, keepdim=True).detach()
        # x_enc = x_enc - means
        # stdev = torch.sqrt(
        #     torch.var(x_enc, dim=2, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev
        # x_enc = rearrange(x_enc, 'b n s m -> b (n s) m')

        # enc_out = torch.nn.functional.pad(x_enc, (0, 768-x_enc.shape[-1]))   # out_layer = (d_ff*window_size)
        input_x = rearrange(x_enc, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x) # （b, m, l+s）
        # print(input_x.shape)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride) # （b，m, n, p）
        # print(input_x.shape)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)') 
        # print(input_x.shape)
        
        enc_out = self.enc_embedding(input_x, None)  # (b, n, d) out_layer = (d_ff* patch_num)
        # print(enc_out.shape)
        outputs = self.gpt2(inputs_embeds=enc_out).last_hidden_state # (b, n, d)
        # print(outputs.shape)

        # outputs = outputs[:, :, :self.d_ff].reshape((outputs.shape[0],-1))
        outputs = outputs[:, :, :self.d_ff]
        # outputs = self.ln_proj(outputs)
        dec_out = self.out_layer(outputs)
        dec_out = torch.mean(dec_out, dim=1)
    

        return dec_out
    



