# Copyright (c) 2022 Xingyu Chen. All Rights Reserved.
# Modified by WangYiteng on 2024-05-20

"""
 * @file modules.py
 * @author chenxingyu (chenxy.sean@gmail.com)
 * @brief Modules composing MobRecon
 * @version 0.1
 * @date 2022-04-28
 *
 * @copyright Copyright (c) 2022 chenxingyu
 *
 * @modified by WangYiteng (2978558373@qq.com)
 * @brief Refactored module structure
 * @version 0.2
 * @date 2024-05-20
 *
"""


import torch.nn as nn
import torch
from conv import DSConv
from LSG.models.GCN import GCNfction
from LSG.models.MHA import MultiHeadAttention
# Basic modules

class Reorg(nn.Module):
    dump_patches = True

    def __init__(self):
        """Reorg layer to re-organize spatial dim and channel dim
        """
        super(Reorg, self).__init__()

    def forward(self, x):
        ss = x.size()
        out = x.view(ss[0], ss[1], ss[2] // 2, 2, ss[3]).view(ss[0], ss[1], ss[2] // 2, 2, ss[3] // 2, 2). \
            permute(0, 1, 3, 5, 2, 4).contiguous().view(ss[0], -1, ss[2] // 2, ss[3] // 2)
        return out


def conv_layer(channel_in, channel_out, ks=1, stride=1, padding=0, dilation=1, bias=False, bn=True, relu=True, group=1):
    """Conv block

    Args:
        channel_in (int): input channel size
        channel_out (int): output channel size
        ks (int, optional): kernel size. Defaults to 1.
        stride (int, optional): Defaults to 1.
        padding (int, optional): Defaults to 0.
        dilation (int, optional): Defaults to 1.
        bias (bool, optional): Defaults to False.
        bn (bool, optional): Defaults to True.
        relu (bool, optional): Defaults to True.
        group (int, optional): group conv parameter. Defaults to 1.

    Returns:
        Sequential: a block with bn and relu
    """
    _conv = nn.Conv2d
    sequence = [_conv(channel_in, channel_out, kernel_size=ks, stride=stride, padding=padding, dilation=dilation,
                      bias=bias, groups=group)]
    if bn:
        sequence.append(nn.BatchNorm2d(channel_out))
    if relu:
        sequence.append(nn.ReLU())

    return nn.Sequential(*sequence)


def linear_layer(channel_in, channel_out, bias=False, bn=True, relu=True):
    """Fully connected block

    Args:
        channel_in (int): input channel size
        channel_out (_type_): output channel size
        bias (bool, optional): Defaults to False.
        bn (bool, optional): Defaults to True.
        relu (bool, optional): Defaults to True.

    Returns:
        Sequential: a block with bn and relu
    """
    _linear = nn.Linear
    sequence = [_linear(channel_in, channel_out, bias=bias)]

    if bn:
        sequence.append(nn.BatchNorm1d(channel_out))
    if relu:
        sequence.append(nn.Hardtanh(0,4))

    return nn.Sequential(*sequence)


class mobile_unit(nn.Module):
    dump_patches = True

    def __init__(self, channel_in, channel_out, stride=1, has_half_out=False, num3x3=1):
        """Init a depth-wise sparable convolution

        Args:
            channel_in (int): input channel size
            channel_out (_type_): output channel size
            stride (int, optional): conv stride. Defaults to 1.
            has_half_out (bool, optional): whether output intermediate result. Defaults to False.
            num3x3 (int, optional): amount of 3x3 conv layer. Defaults to 1.
        """
        super(mobile_unit, self).__init__()
        self.stride = stride
        self.channel_in = channel_in
        self.channel_out = channel_out
        if num3x3 == 1:
            self.conv3x3 = nn.Sequential(
                conv_layer(channel_in, channel_in, ks=3, stride=stride, padding=1, group=channel_in),
            )
        else:
            self.conv3x3 = nn.Sequential(
                conv_layer(channel_in, channel_in, ks=3, stride=1, padding=1, group=channel_in),
                conv_layer(channel_in, channel_in, ks=3, stride=stride, padding=1, group=channel_in),
            )
        self.conv1x1 = conv_layer(channel_in, channel_out)
        self.has_half_out = has_half_out

    def forward(self, x):
        half_out = self.conv3x3(x)
        out = self.conv1x1(half_out)
        if self.stride == 1 and (self.channel_in == self.channel_out):
            out = out + x
        if self.has_half_out:
            return half_out, out
        else:
            return out


def Pool(x, trans, dim=1):
    """Upsample a mesh

    Args:
        x (tensor): input tensor, BxNxD
        trans (tuple): upsample indices and valus
        dim (int, optional): upsample axis. Defaults to 1.

    Returns:
        tensor: upsampled tensor, BxN'xD
    """
    row, col, value = trans[0].to(x.device), trans[1].to(x.device), trans[2].to(x.device)
    value = value.unsqueeze(-1)
    out = torch.index_select(x, dim, col) * value
    out2 = torch.zeros(x.size(0), row.size(0)//3, x.size(-1)).to(x.device)
    idx = row.unsqueeze(0).unsqueeze(-1).expand_as(out)
    out2 = torch.scatter_add(out2, dim, idx, out)
    return out2


class SpiralDeblock(nn.Module):
    def __init__(self, in_channels, out_channels, indices, meshconv= DSConv):
        """Init a spiral conv block

        Args:
            in_channels (int): input feature dim
            out_channels (int): output feature dim
            indices (tensor): neighbourhood of each hand vertex
            meshconv (optional): conv method. Defaults to DSConv.
        """
        super(SpiralDeblock, self).__init__()
        self.conv = meshconv(in_channels, out_channels, indices)
        self.relu = nn.ReLU(inplace=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, up_transform):
        out = Pool(x, up_transform)
        out = self.relu(self.conv(out))
        return out

# Advanced modules
class Reg2DDecode3D(nn.Module):
    def __init__(self, latent_size, out_channels, spiral_indices, up_transform, uv_channel, meshconv=DSConv):
        """Init a 3D decoding with sprial convolution

        Args:
            latent_size (int): feature dim of backbone feature
            out_channels (list): feature dim of each spiral layer
            spiral_indices (list): neighbourhood of each hand vertex
            up_transform (list): upsampling matrix of each hand mesh level
            uv_channel (int): amount of 2D landmark 
            meshconv (optional): conv method. Defaults to DSConv.
        """
        super(Reg2DDecode3D, self).__init__()
        self.latent_size = latent_size
        self.out_channels = out_channels
        self.spiral_indices = spiral_indices
        self.up_transform = up_transform
        self.num_vert = [u[0].size(0)//3 for u in self.up_transform] + [self.up_transform[-1][0].size(0)//6]
        self.uv_channel = uv_channel
        self.de_layer_conv = conv_layer(self.latent_size, self.out_channels[- 1], 1, bn=False, relu=False)
        self.de_layer = nn.ModuleList()
        for idx in range(len(self.out_channels)):
            if idx == 0:
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx - 1], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))
            else:
                self.de_layer.append(SpiralDeblock(self.out_channels[-idx], self.out_channels[-idx - 1], self.spiral_indices[-idx - 1], meshconv=meshconv))
        self.head = meshconv(self.out_channels[0], 3, self.spiral_indices[0])
        self.upsample = nn.Parameter(torch.ones([self.num_vert[-1], self.uv_channel])*0.01, requires_grad=True)
        self.verts = [49, 98, 195, 389]
        self.dim = [256, 256, 128, 64]
        self.GCNlist=nn.ModuleList()
        for i in range(4):
            self.GCNlist.append(GCNfction(self.verts[i],self.dim[i]))
        self.MultiHead = MultiHeadAttention(256, 4, num_layers=4, dropout_rate=0.0)
        self.conv1x1 = nn.Conv1d(in_channels=688, out_channels=256, kernel_size=1)
    def index(self, feat, uv):
        uv = uv.unsqueeze(2)  # [B, N, 1, 2]
        samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
        return samples[:, :, :, 0]  # [B, C, N]

    def forward(self, uv, x,pre_out,stack1_out,stack2_out):
        uv = torch.clamp((uv - 0.5) * 2, -1, 1)
        x = self.de_layer_conv(x)
        #局部特征采样阶段
        pre_out_feat = self.index(pre_out, uv).permute(0, 2, 1)
        stack1_out_feat = self.index(stack1_out, uv).permute(0, 2, 1)
        stack2_out_feat = self.index(stack2_out, uv).permute(0, 2, 1)

        #主要特征采样阶段
        latent_out = self.index(x, uv).permute(0, 2, 1)
        #自注意阶段
        attention_out = self.MultiHead(latent_out)
        #特征拼接
        cat_feat = torch.cat((pre_out_feat, stack1_out_feat , stack2_out_feat, latent_out, attention_out), dim=2)

        #主要粗糙网格
        x = torch.bmm(self.upsample.repeat(attention_out.size(0), 1, 1).to(attention_out.device), attention_out)
        num_features = len(self.de_layer)
        for i, layer in enumerate(self.de_layer):
            x_GCN = self.GCNlist[i](x)
            x = layer(x_GCN, self.up_transform[num_features - i - 1])

        pred_rough = self.head(x)

        #次要精细化量
        y = cat_feat.permute(0, 2, 1)
        y= self.conv1x1(y)
        y= y.permute(0, 2, 1)
        y= torch.bmm(self.upsample.repeat(y.size(0), 1, 1).to(y.device), y)

        num_features = len(self.de_layer)
        for i, layer in enumerate(self.de_layer):
            y_GCN = self.GCNlist[i](y)
            y = layer(y_GCN, self.up_transform[num_features - i - 1])

        pred_fine = self.head(y)


        return pred_rough,pred_rough+pred_fine
