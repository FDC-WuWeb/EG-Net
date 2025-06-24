import numpy as np
import torch
import torch.nn as nn
from IPython import embed
from einops import rearrange, repeat
from geotransformer.modules.kpconv import ConvBlock, ResidualBlock, UnaryBlock, LastUnaryBlock, nearest_upsample
import ops
import math
import open3d as o3d
import numpy as np
class EdgeAttention(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(EdgeAttention, self).__init__()
        self.heads = 2
        self.K = 1
        self.group_type = 'diff'
        self.q_conv = nn.Conv2d(inchannel, outchannel, 1, bias=False)
        self.k_conv = nn.Conv2d(inchannel, outchannel, 1, bias=False)
        self.v_conv = nn.Conv2d(inchannel, outchannel, 1, bias=False)
        self.c_conv = nn.Conv2d(inchannel, outchannel, 1, bias=False)
        self.c_conv2 = nn.Conv2d(outchannel, inchannel, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Conv1d(outchannel, outchannel*2, 1, bias=False), nn.LeakyReLU(0.2), nn.Conv1d(outchannel*2, outchannel, 1, bias=False))
        self.bn1 = nn.BatchNorm1d(outchannel)
        self.bn2 = nn.BatchNorm1d(outchannel)

    def forward(self, x):
        
        neighbors = ops.group(x, self.K, self.group_type)  # (B, C, N) -> (B, C, N, K)
        q = self.q_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()  # (B, C, N) -> (B, C, N, 1)
        q = self.split_heads(q, self.heads)  # (B, C, N, 1) -> (B, H, N, 1, D)
        k = self.k_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        k = self.split_heads(k, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        v = self.v_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        v = self.split_heads(v, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)
        energy = q @ rearrange(k, 'B H N K D -> B H N D K').contiguous()  # (B, H, N, 1, D) @ (B, H, N, D, K) -> (B, H, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # (B, H, N, 1, K) -> (B, H, N, 1, K)
        tmp = rearrange(attention@v, 'B H N 1 D -> B (H D) N').contiguous()  # (B, H, N, 1, K) @ (B, H, N, K, D) -> (B, H, N, 1, D) -> (B, C=H*D, N)

        x = self.c_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()
        x = torch.squeeze(x, 3)

        x = self.bn1(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        tmp = self.ff(x)  # (B, C, N) -> (B, C, N)
        x = self.bn2(x + tmp)  # (B, C, N) + (B, C, N) -> (B, C, N)
        # x = torch.unsqueeze(x, 3)
        # x = self.c_conv2(x)
        return x
    @staticmethod
    def split_heads(x, heads):
        x = rearrange(x, 'B (H D) N K -> B H N K D', H=heads).contiguous()  # (B, C, N, K) -> (B, H, N, K, D)
        return x

class GlobalDownSample(nn.Module):
    def __init__(self, channels):
        super(GlobalDownSample, self).__init__()
        # self.npts_ds = npts_ds
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, npts_ds):
        q = self.q_conv(x)  # (B, C, N) -> (B, C, N)
        k = self.k_conv(x)  # (B, C, N) -> (B, C, N)
        v = self.v_conv(x)  # (B, C, N) -> (B, C, N)
        energy = rearrange(q, 'B C N -> B N C').contiguous() @ k  # (B, N, C) @ (B, C, N) -> (B, N, N)
        scale_factor = math.sqrt(q.shape[-2])
        attention = self.softmax(energy / scale_factor)  # (B, N, N) -> (B, N, N)
        selection = torch.sum(attention, dim=-2)  # (B, N, N) -> (B, N)
        self.idx = selection.topk(npts_ds, dim=-1)[1]  # (B, N) -> (B, M)
        scores = torch.gather(attention, dim=1, index=repeat(self.idx, 'B M -> B M N', N=attention.shape[-1]))  # (B, N, N) -> (B, M, N)
        v = scores @ rearrange(v, 'B C N -> B N C').contiguous()  # (B, M, N) @ (B, N, C) -> (B, M, C)
        out = rearrange(v, 'B M C -> B C M').contiguous()  # (B, M, C) -> (B, C, M)
        return out


class LocalDownSample(nn.Module):
    def __init__(self, channels):
        super(LocalDownSample, self).__init__()
        # self.npts_ds = npts_ds  # number of downsampled points
        self.K = 4  # number of neighbors
        self.group_type = 'diff'
        self.q_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, npts_ds):
        neighbors = ops.group(x, self.K, self.group_type)  # (B, C, N) -> (B, C, N, K)
        q = self.q_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()  # (B, C, N) -> (B, C, N, 1)
        q = rearrange(q, 'B C N 1 -> B N 1 C').contiguous()  # (B, C, N, 1) -> (B, N, 1, C)
        k = self.k_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        k = rearrange(k, 'B C N K -> B N C K').contiguous()  # (B, C, N, K) -> (B, N, C, K)
        v = self.v_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        v = rearrange(v, 'B C N K -> B N K C').contiguous()  # (B, C, N, K) -> (B, N, K, C)
        energy = q @ k  # (B, N, 1, C) @ (B, N, C, K) -> (B, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # (B, N, 1, K) -> (B, N, 1, K)
        selection = rearrange(torch.std(attention, dim=-1, unbiased=False), 'B N 1 -> B N').contiguous()  # (B, N, 1, K) -> (B, N, 1) -> (B, N)
        self.idx = selection.topk(npts_ds, dim=-1)[1]  # (B, N) -> (B, M)
        scores = torch.gather(attention, dim=1, index=repeat(self.idx, 'B M -> B M 1 K', K=attention.shape[-1]))  # (B, N, 1, K) -> (B, M, 1, K)
        v = torch.gather(v, dim=1, index=repeat(self.idx, 'B M -> B M K C', K=v.shape[-2], C=v.shape[-1]))  # (B, N, K, C) -> (B, M, K, C)
        out = rearrange(scores@v, 'B M 1 C -> B C M').contiguous()  # (B, M, 1, K) @ (B, M, K, C) -> (B, M, 1, C) -> (B, C, M)
        return out

class KPConvFPN(nn.Module):
    def __init__(self, input_dim, output_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm):
        super(KPConvFPN, self).__init__()

        self.encoder1_1 = ConvBlock(input_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm)
        self.encoder1_2 = ResidualBlock(init_dim, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm)

        self.encoder2_1 = ResidualBlock(
            init_dim * 2, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm, strided=True
        )
        self.encoder2_2 = ResidualBlock(
            init_dim * 2, init_dim * 2, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )

        self.encoder3_1 = ResidualBlock(
            init_dim * 2, init_dim * 2, kernel_size, init_radius * 2, init_sigma * 2, group_norm, strided=True
        )

        self.encoder3_2 = ResidualBlock(
            init_dim * 2, init_dim * 2, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )

        self.encoder4_1 = ResidualBlock(
            init_dim * 2, init_dim * 4, kernel_size, init_radius * 4, init_sigma * 4, group_norm, strided=True
        )

        self.encoder4_2 = ResidualBlock(
            init_dim * 4, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, group_norm
        )

        self.decoder3 = UnaryBlock(init_dim * 18, init_dim * 2, group_norm)
        self.decoder2 = LastUnaryBlock(init_dim * 4, output_dim)


        p1,p2,p3 = 8,16,24
        self.n2p_attention1 = EdgeAttention(3, p1)
        self.n2p_attention2 = EdgeAttention(3, p2)
        self.n2p_attention3 = EdgeAttention(3, p3)
        self.LDS1 = LocalDownSample(p1)
        self.GDS1 = GlobalDownSample(p1)
        self.linear_layer1 = nn.Linear(p1, 3)
        self.LDS2 = LocalDownSample(p2)
        self.GDS2 = GlobalDownSample(p2)
        self.linear_layer2 = nn.Linear(p2, 3)
        self.LDS3 = LocalDownSample(p3)
        self.GDS3 = GlobalDownSample(p3)
        self.linear_layer3 = nn.Linear(p3, 3)


    def forward(self, feats, data_dict):
        feats_list = []

        points_list = data_dict['points']
        neighbors_list = data_dict['neighbors']
        subsampling_list = data_dict['subsampling']
        upsampling_list = data_dict['upsampling']
       
        feats_s1 = feats

        feats_s1 = self.encoder1_1(feats_s1, points_list[0], points_list[0], neighbors_list[0])
        feats_s1 = self.encoder1_2(feats_s1, points_list[0], points_list[0], neighbors_list[0])

        points_list[0] = torch.unsqueeze(points_list[0], 0).permute(0, 2, 1) # torch.Size([1, 3, 4000])
        points_list[1] = torch.unsqueeze(points_list[1], 0).permute(0, 2, 1) # torch.Size([1, 3, _])
        points_list[0] = self.n2p_attention1(points_list[0]) # torch.Size([1, 256, 4000])
        points_list[1] = self.LDS1(points_list[0],points_list[1].shape[2])  # torch.Size([1, 256, _])
        # points_list[1] = self.GDS1(points_list[0],points_list[1].shape[2])  # torch.Size([1, 256, _])
        points_list[1] = torch.squeeze(points_list[1]).permute(1, 0)  # torch.Size([_, 256])
        points_list[1] = self.linear_layer1(points_list[1])
        points_list[0] = torch.squeeze(points_list[0]).permute(1, 0)
        points_list[0] = self.linear_layer1(points_list[0])

        # print("down")
        feats_s2 = self.encoder2_1(feats_s1, points_list[1], points_list[0], subsampling_list[0])
        feats_s2 = self.encoder2_2(feats_s2, points_list[1], points_list[1], neighbors_list[1])
        # input()

        points_list[1] = torch.unsqueeze(points_list[1], 0).permute(0, 2, 1) # torch.Size([1, 3, 4000])
        points_list[2] = torch.unsqueeze(points_list[2], 0).permute(0, 2, 1) # torch.Size([1, 3, _])
        points_list[1] = self.n2p_attention2(points_list[1]) # torch.Size([1, 256, 4000])
        points_list[2] = self.LDS2(points_list[1],points_list[2].shape[2])  # torch.Size([1, 256, _])
        points_list[2] = torch.squeeze(points_list[2]).permute(1, 0)  # torch.Size([_, 256])
        points_list[2] = self.linear_layer2(points_list[2])
        points_list[1] = torch.squeeze(points_list[1]).permute(1, 0)
        points_list[1] = self.linear_layer2(points_list[1])

        feats_s3 = self.encoder3_1(feats_s2, points_list[2], points_list[1], subsampling_list[1])
        feats_s3 = self.encoder3_2(feats_s3, points_list[2], points_list[2], neighbors_list[2])


        points_list[2] = torch.unsqueeze(points_list[2], 0).permute(0, 2, 1) # torch.Size([1, 3, 4000])
        points_list[3] = torch.unsqueeze(points_list[3], 0).permute(0, 2, 1) # torch.Size([1, 3, _])
        points_list[2] = self.n2p_attention3(points_list[2]) # torch.Size([1, 256, 4000])
        points_list[3] = self.LDS3(points_list[2],points_list[3].shape[2])  # torch.Size([1, 256, _])
        points_list[3] = torch.squeeze(points_list[3]).permute(1, 0)  # torch.Size([_, 256])
        points_list[3] = self.linear_layer3(points_list[3])
        points_list[2] = torch.squeeze(points_list[2]).permute(1, 0)
        points_list[2] = self.linear_layer3(points_list[2])

        feats_s4 = self.encoder4_1(feats_s3, points_list[3], points_list[2], subsampling_list[2])
        feats_s4 = self.encoder4_2(feats_s4, points_list[3], points_list[3], neighbors_list[3])

        latent_s4 = feats_s4
        feats_list.append(feats_s4)

        latent_s3 = nearest_upsample(latent_s4, upsampling_list[2])
        latent_s3 = torch.cat([latent_s3, feats_s3], dim=1)
        latent_s3 = self.decoder3(latent_s3)


        feats_list.append(latent_s3)

        latent_s2 = nearest_upsample(latent_s3, upsampling_list[1])
        latent_s2 = torch.cat([latent_s2, feats_s2], dim=1)
        latent_s2 = self.decoder2(latent_s2)


        feats_list.append(latent_s2)
        feats_list.reverse()

        return feats_list
