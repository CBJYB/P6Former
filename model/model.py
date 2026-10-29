import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class SpatialEncoder(nn.Module):
    def __init__(self, in_vars=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_vars, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),             # -> 40x70
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),             # -> 20x35
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)) # -> 8x8
        )
        self.out_dim = 128 * 8 * 8

    def forward(self, x):
        return self.encoder(x).reshape(x.size(0), -1)

class P6Former(nn.Module):
    def __init__(self, time_steps, in_vars, num_indices, d_model, n_heads,
                 num_layers, d_ff, dropout, spatial_size):
        super().__init__()
        self.time_steps = time_steps
        self.in_vars = in_vars
        self.num_indices = num_indices
        self.d_model = d_model
        H, W = spatial_size
        self.H, self.W = H, W

        #  空间编码
        self.spatial_encoder = SpatialEncoder(in_vars)
        self.flatten_dim = self.spatial_encoder.out_dim
        self.input_proj = nn.Linear(self.flatten_dim, d_model)

        # 时间编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=time_steps)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 主输出头：全连接出 8x8，再用反卷积放大
        self.fc_main = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_indices * 8 * 8)
        )
        self.deconv_main = nn.Sequential(
            nn.ConvTranspose2d(num_indices, 32, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),           # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_indices, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.Upsample(size=(self.H, self.W), mode='bilinear', align_corners=False)  # 64x64 -> 80x140
        )

        #辅助输出头
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, in_vars * 8 * 8)
        )
        self.deconv_aux = nn.Sequential(
            nn.ConvTranspose2d(in_vars, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_vars, kernel_size=4, stride=2, padding=1),
            nn.Upsample(size=(self.H, self.W), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        feats = []
        for t in range(T):
            feats.append(self.spatial_encoder(x[:, t, :, :, :]))
        feats = torch.stack(feats, dim=1)

        encoded = self.input_proj(feats)
        encoded = self.pos_encoder(encoded)
        encoded = self.transformer(encoded)

        global_feat = encoded.mean(dim=1)

        # 主输出：全连接 -> 反卷积
        main_low = self.fc_main(global_feat).view(B, self.num_indices, 8, 8)
        main_out = self.deconv_main(main_low)

        # 辅助输出：全连接 -> 反卷积
        aux_low = self.decoder(global_feat).view(B, self.in_vars, 8, 8)
        aux_out = self.deconv_aux(aux_low)

        return main_out, aux_out