# Copyright (c) 2024 WangYiteng and LiMinqi. All Rights Reserved.

"""
 * @file MHSA.py
 * @author WangYiteng and LiMinqi (2978558373@qq.com minqi.li@xpu.edu.cn)
 * @brief Description of what the file does
 * @version 1.0
 * @date 2024-05-20
 *
 * @copyright Copyright (c) 2024 Wang Yiteng and LiMinqi. All Rights Reserved.
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
        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)
        self.output_linear = nn.Linear(input_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, input_dim))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_length = x.size(1)
        x = x + self.positional_encoding[:, :seq_length, :]
        residual = x
        x = self.layer_norm(x)
        query = self.query_linear(x)
        key = self.key_linear(x)
        value = self.value_linear(x)
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.input_dim)
        output = self.output_linear(context)
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
    input_dim = 256
    num_heads = 4
    seq_length = 21
    batch_size = 10
    num_layers = 4
    multi_head_attention = MultiHeadAttention(input_dim, num_heads, num_layers=num_layers, dropout_rate=0.1)
    x = torch.randn(batch_size, seq_length, input_dim)
    output = multi_head_attention(x)

    print(count_parameters(multi_head_attention)/1000000)
    print(output.shape)
