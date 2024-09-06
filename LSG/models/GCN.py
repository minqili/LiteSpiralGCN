# Copyright (c) 2024 Wang Yiteng. All Rights Reserved.

"""
 * @file GCN.py
 * @author Wang Yiteng (2978558373@qq.com)
 * @brief Description of what the file does
 * @version 1.0
 * @date 2024-05-20
 *
 * @copyright Copyright (c) 2024 Wang Yiteng. All Rights Reserved.
 *
"""

import torch.nn as nn
import torch

class AGCN(nn.Module):
    def __init__(self, num_joint, features,):
        super(AGCN, self).__init__()
        self.fc = nn.Linear(in_features=features, out_features=features)
        self.adj = nn.Parameter(torch.eye(num_joint).float(), requires_grad=True)

    def laplacian(self, A_hat):
        D_hat = torch.sum(A_hat, 1, keepdim=True) + 1e-5
        L = 1 / D_hat * A_hat
        return L

    def forward(self, x):
        batch = x.size(0)
        A_hat = self.laplacian(self.adj)
        A_hat = A_hat.unsqueeze(0).repeat(batch, 1, 1)
        out = self.fc(torch.matmul(A_hat, x))
        return out

class GCNfction(nn.Module):
    def __init__(self, num_joint, features):
        super(GCNfction, self).__init__()
        self.out = nn.Sequential(
            torch.nn.LayerNorm(features),
            AGCN(num_joint, features),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            AGCN(num_joint, features),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        out = self.out(x)+x
        return out

if __name__ == '__main__':
    """Test the model"""
    model = GCNfction(21, 128)
    model_out = model(torch.zeros(2, 21, 128))
    print(model_out.size())
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters: ", total_params / 1000000)
