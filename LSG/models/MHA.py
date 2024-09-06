# Copyright (c) 2024 Wang Yiteng. All Rights Reserved.

"""
 * @file MHA.py
 * @author Wang Yiteng (2978558373@qq.com)
 * @brief Description of what the file does
 * @version 1.0
 * @date 2024-05-20
 *
 * @copyright Copyright (c) 2024 Wang Yiteng. All Rights Reserved.
 *
"""



import torch
import torch.nn as nn

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dropout_rate=0.0):
        super(MultiHeadAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        assert input_dim % num_heads == 0, "输入维度必须可以被头的数量整除."

        # 为每个头创建查询、键和值的线性层
        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)

        # 输出线性层在连接多头输出后
        self.output_linear = nn.Linear(input_dim, input_dim)

        # 头归一化层
        self.layer_norm = nn.LayerNorm(input_dim)

        # 学习的位置编码参数
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, input_dim))

        # Dropout 层用于正则化
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_length = x.size(1)

        # 将学习的位置编码添加到输入中
        x = x + self.positional_encoding[:, :seq_length, :]

        # 自注意力计算之前进行层归一化，并添加残差连接
        residual = x
        x = self.layer_norm(x)

        # 线性变换得到查询、键和值
        query = self.query_linear(x)
        key = self.key_linear(x)
        value = self.value_linear(x)

        # 将查询、键和值重塑为多头形式
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算每个头的缩放点积注意力
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        attention_probs = torch.softmax(attention_scores, dim=-1)

        # 应用 dropout
        attention_probs = self.dropout(attention_probs)

        # 使用注意力概率对值加权求和得到上下文向量
        context = torch.matmul(attention_probs, value)

        # 将多头连接起来
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.input_dim)

        # 输出线性变换
        output = self.output_linear(context)

        # 添加残差连接
        output += residual

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers=1, dropout_rate=0.0):
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.attention_layers = nn.ModuleList([
            MultiHeadAttentionLayer(input_dim, num_heads, dropout_rate) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.attention_layers:
            x = layer(x, mask)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 示例用法:
    input_dim = 256
    num_heads = 4
    seq_length = 21
    batch_size = 10
    num_layers = 4 # 你可以在这里更改层数

    # 创建多头自注意力层，可选择层数
    multi_head_attention = MultiHeadAttention(input_dim, num_heads, num_layers=num_layers, dropout_rate=0.1)

    # 随机输入张量
    x = torch.randn(batch_size, seq_length, input_dim)

    # 计算多头自注意力
    output = multi_head_attention(x)

    print("模型中的总参数数量:", count_parameters(multi_head_attention)/1000000)
    print("输出形状:", output.shape)
