from torch import nn
import torch


class TCN(nn.Module):
    def __init__(self, configs):
        super(TCN, self).__init__()
        self.input_channels = configs.input_channels
        self.final_out_channels = configs.final_out_channels
        self.features_len = configs.features_len
        self.project_channels = configs.project_channels
        self.hidden_size = configs.hidden_size
        self.window_size = configs.window_size
        self.num_layers = configs.num_layers
        self.kernel_size = configs.kernel_size
        self.stride = configs.stride
        self.dropout = configs.dropout
        self.d_ff = configs.d_ff
        
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(self.input_channels, 32, kernel_size=self.kernel_size,
                      stride=self.stride, bias=False, padding=(self.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(self.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, self.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(self.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )


        self.encoder = nn.LSTM(
            self.final_out_channels,
            self.final_out_channels,
            batch_first=True,
            num_layers=self.num_layers,
            bias=False,
            dropout=self.dropout,
        )


        self.output_layer = nn.Linear(self.d_ff*self.window_size, self.final_out_channels)

        self.projection_head = nn.Sequential(
            nn.Linear(self.final_out_channels * self.features_len, self.final_out_channels * self.features_len // 2),
            nn.BatchNorm1d(self.final_out_channels * self.features_len // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.final_out_channels * self.features_len // 2, self.project_channels),
        )

        self.gatenet = nn.Sequential(
            nn.Linear(2*self.input_channels, 2 * self.project_channels),
            nn.BatchNorm1d(2 * self.project_channels),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.project_channels, self.project_channels))
        self.statnet = nn.Sequential(
            nn.Linear(2*self.input_channels, 2 * self.project_channels),
            nn.BatchNorm1d(2 * self.project_channels),
            nn.ReLU(inplace=True),
            nn.Linear(2 * self.project_channels, self.project_channels))
    
    def forward(self, x_in, gate=None):
        if not gate:
            x = self.conv_block1(x_in)
            x = self.conv_block2(x)
            # Encoder
            hidden = x.permute(0, 2, 1) # (B, t, d)
        
            output, (enc_hidden, enc_cell) = self.encoder(hidden)
            '''output: (B, t, D)
            enc_hidden: (num_layer, B, D)
            enc_cell: (num_layer, B,  D)'''
            
            output = output.reshape(output.size(0), -1)
            project = self.projection_head(output)

            return project

        else:
            '''x: (B, T, d)'''
            means = x_in.mean(2, keepdim=True).detach()
            x_in = x_in - means  # (B, 1, d)
            stdev = torch.sqrt(
                torch.var(x_in, dim=2, keepdim=True, unbiased=False) + 1e-5) # (B, 1, d)
            x_in /= stdev

            x_stat = torch.cat((means, stdev), dim=1).squeeze(2) # (B, 2d)


            x_gate = torch.sigmoid(self.gatenet(x_stat))
            x_scale = self.statnet(x_stat)

            
            x = self.conv_block1(x_in)
            x = self.conv_block2(x)
            # Encoder
            hidden = x.permute(0, 2, 1) # (B, t, d)
        
            output, (enc_hidden, enc_cell) = self.encoder(hidden)
            '''output: (B, t, D)
            enc_hidden: (num_layer, B, D)
            enc_cell: (num_layer, B,  D)'''
            
            output = output.reshape(output.size(0), -1)
            output = self.projection_head(output)
            
            project = output * (1 - x_gate) + x_gate * x_scale
            
            return project