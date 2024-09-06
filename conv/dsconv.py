# Copyright (c) Xingyu Chen. All Rights Reserved.

"""
 * @file dsconv.py
 * @author chenxingyu (chenxy.sean@gmail.com)
 * @brief Depth-separable spiral convolution
 * @version 0.1
 * @date 2022-04-28
 *
 * @copyright Copyright (c) 2022 chenxingyu
 *
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# class DSConv(nn.Module):
#     def __init__(self, in_channels, out_channels, indices, dim=1):
#         super(DSConv, self).__init__()
#         self.dim = dim
#         self.indices = indices
#         self.n_nodes, _ = self.indices.size()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.seq_length = indices.size(1)
#         self.spatial_layer = nn.Conv2d(self.in_channels, self.in_channels, int(np.sqrt(self.seq_length)), 1, 0, groups=self.in_channels, bias=False)
#         self.channel_layer = nn.Linear(self.in_channels, self.out_channels, bias=False)
#         torch.nn.init.xavier_uniform_(self.channel_layer.weight)
#         self.fc_loc = nn.Sequential(
#             nn.Linear(self.out_channels, 128),
#             nn.ReLU(True),
#             nn.Linear(128, 64),
#             nn.ReLU(True),
#             nn.Linear(64, 6)
#         )
#         self.LN = nn.LayerNorm(self.in_channels)
#     def reset_parameters(self):
#         torch.nn.init.xavier_uniform_(self.spatial_layer.weight)
#         torch.nn.init.xavier_uniform_(self.channel_layer.weight)
#
#
#     def forward(self, x):
#         n_nodes, _ = self.indices.size()
#         bs = x.size(0)
#         x = self.LN(x)
#         x = torch.index_select(x, self.dim, self.indices.to(x.device).view(-1))
#         x = x.view(bs * n_nodes, self.seq_length, -1).transpose(1, 2)
#         x = x.view(x.size(0), x.size(1), int(np.sqrt(self.seq_length)), int(np.sqrt(self.seq_length)))
#
#         xs = self.spatial_layer(x).view(bs, self.n_nodes, -1)
#         xs = self.channel_layer(xs)
#         xs = xs.view(bs * n_nodes, -1)
#         theta = self.fc_loc(xs)
#         theta = theta.view(-1, 2, 3)
#         grid = F.affine_grid(theta, x.size())
#         x = F.grid_sample(x, grid)
#
#         x = self.spatial_layer(x).view(bs,n_nodes, -1)
#         x = self.channel_layer(x)
#
#
#
#
#         return x
#
#     def __repr__(self):
#         return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
#                                                   self.in_channels,
#                                                   self.out_channels,
#                                                   self.seq_length)



# class DSConv(nn.Module):
#     def __init__(self, in_channels, out_channels, indices, dim=1):
#         super(DSConv, self).__init__()
#         self.dim = dim
#         self.indices = indices
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.seq_length = indices.size(1)
#         self.spatial_layer = nn.Conv2d(self.in_channels, self.in_channels, int(np.sqrt(self.seq_length)), 1, 0, groups=self.in_channels, bias=False)
#         self.channel_layer = nn.Linear(self.in_channels, self.out_channels, bias=False)
#         torch.nn.init.xavier_uniform_(self.channel_layer.weight)
#         # Regressor for the 3 * 2 affine matrix
#         self.fc_loc = nn.Sequential(
#             nn.Linear(10 * 3 * 3, 32),
#             nn.ReLU(True),
#             nn.Linear(32, 3 * 2)
#         )
#
#         # Initialize the weights/bias with identity transformation
#         self.fc_loc[2].weight.data.zero_()
#         self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
#         # Spatial transformer localization-network
#         self.localization = nn.Sequential(
#             nn.Conv2d(self.in_channels, 20, kernel_size=1),
#
#             nn.ReLU(True),
#             nn.Conv2d(20, 10, kernel_size=1),
#
#             nn.ReLU(True)
#         )
#     def reset_parameters(self):
#         torch.nn.init.xavier_uniform_(self.spatial_layer.weight)
#         torch.nn.init.xavier_uniform_(self.channel_layer.weight)
#
#     def stn(self, x):
#         xs = self.localization(x)
#         xs = xs.contiguous()
#         xs = xs.view(-1,10 * 3 * 3)
#         theta = self.fc_loc(xs)
#         theta = theta.view(-1, 2, 3)
#
#         grid = F.affine_grid(theta, x.size())
#         x = F.grid_sample(x, grid)
#
#         return x
#
#     def forward(self, x):
#         n_nodes, _ = self.indices.size()
#         bs = x.size(0)
#         x = torch.index_select(x, self.dim, self.indices.to(x.device).view(-1))
#         x = x.view(bs * n_nodes, self.seq_length, -1).transpose(1, 2)
#         x = x.view(x.size(0), x.size(1), int(np.sqrt(self.seq_length)), int(np.sqrt(self.seq_length)))
#         x = self.stn(x)
#         x = self.spatial_layer(x).view(bs, n_nodes, -1)
#         x = self.channel_layer(x)
#
#         return x
#
#
#     def __repr__(self):
#         return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
#                                                   self.in_channels,
#                                                   self.out_channels,
#                                                   self.seq_length)

class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1):
        super(DSConv, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)
        self.spatial_layer = nn.Conv2d(self.in_channels, self.in_channels, int(np.sqrt(self.seq_length)), 1, 0, groups=self.in_channels, bias=False)
        self.channel_layer = nn.Linear(self.in_channels, self.out_channels, bias=False)
        torch.nn.init.xavier_uniform_(self.channel_layer.weight)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.spatial_layer.weight)
        torch.nn.init.xavier_uniform_(self.channel_layer.weight)

    def forward(self, x):
        n_nodes, _ = self.indices.size()
        bs = x.size(0)
        x = torch.index_select(x, self.dim, self.indices.to(x.device).view(-1))
        x = x.view(bs * n_nodes, self.seq_length, -1).transpose(1, 2)
        x = x.view(x.size(0), x.size(1), int(np.sqrt(self.seq_length)), int(np.sqrt(self.seq_length)))
        x = self.spatial_layer(x).view(bs, n_nodes, -1)
        x = self.channel_layer(x)

        return x

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)