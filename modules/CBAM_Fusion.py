import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.channel = nn.Conv2d(in_channels, in_channels, 7, padding=0)

    def forward(self, x):
        x = self.channel(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        return spatial_out


class CBAMFusion(nn.Module):
    def __init__(self, in_channels, reduction=16, spatial_kernel_size=7):
        super(CBAMFusion, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel_size)
        self.x1conv = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self,  x_small, x_large):
        # Step 1: Apply Channel Attention on smaller feature map (256x7x7)
        ca = self.channel_attention(x_small)

        # Step 2: Apply the channel attention weights to the larger feature map
        x_large_ca = x_large * ca

        # Step 3: Apply Spatial Attention on smaller feature map (256x7x7)
        sa = self.spatial_attention(x_small)

        # Step 4: Apply the spatial attention weights to the larger feature map
        sa_upsampled = F.interpolate(sa, size=x_large.shape[2:], mode='bilinear', align_corners=False)
        sa = self.x1conv(sa_upsampled)
        x_large_sa = x_large_ca * sa

        return x_large_sa