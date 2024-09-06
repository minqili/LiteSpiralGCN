# Copyright (c) 2022 Xingyu Chen. All Rights Reserved.
# Modified by WangYiteng on 2024-05-20

"""
 * @file litespiralgcn.py
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

import sys
import os
from thop import  profile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch.nn as nn
import torch
from LSG.models.densestack import DenseStack_Backnone
from LSG.models.modules import Reg2DDecode3D
from LSG.models.loss import l1_loss, normal_loss, edge_length_loss, contrastive_loss_3d, contrastive_loss_2d
from utils.read import spiral_tramsform
from conv.dsconv import DSConv
from LSG.build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class LiteSpiralGCN(nn.Module):
    def __init__(self, cfg):
        """Init a DenseStack model

        Args:
            cfg : config file
        """
        super(LiteSpiralGCN, self).__init__()
        self.cfg = cfg
        self.backbone = DenseStack_Backnone(latent_size=cfg.MODEL.LATENT_SIZE,
                                            kpts_num=cfg.MODEL.KPTS_NUM)
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        template_fp = os.path.join(cur_dir, '../../template/template.ply')
        transform_fp = os.path.join(cur_dir, '../../template', 'transform.pkl')
        spiral_indices, _, up_transform, tmp = spiral_tramsform(transform_fp,
                                                                template_fp,
                                                                cfg.MODEL.SPIRAL.DOWN_SCALE,
                                                                cfg.MODEL.SPIRAL.LEN,
                                                                cfg.MODEL.SPIRAL.DILATION)
        for i in range(len(up_transform)):
            up_transform[i] = (*up_transform[i]._indices(), up_transform[i]._values())
        self.decoder3d = Reg2DDecode3D(cfg.MODEL.LATENT_SIZE, 
                                       cfg.MODEL.SPIRAL.OUT_CHANNELS, 
                                       spiral_indices, 
                                       up_transform, 
                                       cfg.MODEL.KPTS_NUM,
                                       meshconv=(DSConv))

    def forward(self, x):
        if x.size(1) == 6:
            pred3d_list = []
            pred3d_rough_list = []
            pred2d_pt_list = []
            for i in range(2):
                latent, pred2d_pt,pre_out,stack1_out,stack2_out = self.backbone(x[:, 3*i:3*i+3])
                pred3d_rough,pred3d = self.decoder3d(pred2d_pt, latent,pre_out,stack1_out,stack2_out)
                pred3d_list.append(pred3d)
                pred3d_rough_list.append(pred3d_rough)
                pred2d_pt_list.append(pred2d_pt)
            pred2d_pt = torch.cat(pred2d_pt_list, -1)
            pred3d = torch.cat(pred3d_list, -1)
            pred3d_rough = torch.cat(pred3d_rough_list, -1)
        else:
            latent, pred2d_pt,pre_out,stack1_out,stack2_out = self.backbone(x)
            pred3d_rough,pred3d = self.decoder3d(pred2d_pt, latent,pre_out,stack1_out,stack2_out)

        return {'verts': pred3d,
                'verts_rough': pred3d_rough,
                'joint_img': pred2d_pt
                }

    def loss(self, **kwargs):
        loss_dict = dict()
        loss_dict['verts_rough_loss'] = 0.3 * l1_loss(kwargs['verts_rough_pred'], kwargs['verts_gt'])
        loss_dict['verts_loss'] = l1_loss(kwargs['verts_pred'], kwargs['verts_gt'])
        loss_dict['joint_img_loss'] = l1_loss(kwargs['joint_img_pred'], kwargs['joint_img_gt'])
        if self.cfg.DATA.CONTRASTIVE:
            loss_dict['normal_loss'] = 0.05 * (normal_loss(kwargs['verts_pred'][..., :3], kwargs['verts_gt'][..., :3], kwargs['face']) + \
                                               normal_loss(kwargs['verts_pred'][..., 3:], kwargs['verts_gt'][..., 3:], kwargs['face']))
            loss_dict['edge_loss'] = 0.5 * (edge_length_loss(kwargs['verts_pred'][..., :3], kwargs['verts_gt'][..., :3], kwargs['face']) + \
                                            edge_length_loss(kwargs['verts_pred'][..., 3:], kwargs['verts_gt'][..., 3:], kwargs['face']))
            if kwargs['aug_param'] is not None:
                loss_dict['con3d_loss'] = contrastive_loss_3d(kwargs['verts_pred'], kwargs['aug_param'])
                loss_dict['con2d_loss'] = contrastive_loss_2d(kwargs['joint_img_pred'], kwargs['bb2img_trans'], kwargs['size'])
        else:
            loss_dict['normal_loss'] = 0.1 * normal_loss(kwargs['verts_pred'], kwargs['verts_gt'], kwargs['face'].to(kwargs['verts_pred'].device))
            loss_dict['edge_loss'] = edge_length_loss(kwargs['verts_pred'], kwargs['verts_gt'], kwargs['face'].to(kwargs['verts_pred'].device))

        loss_dict['loss'] = loss_dict.get('verts_loss', 0) \
                            + loss_dict.get('verts_rough_loss', 0) \
                            + loss_dict.get('normal_loss', 0) \
                            + loss_dict.get('edge_loss', 0) \
                            + loss_dict.get('joint_img_loss', 0) \
                            + loss_dict.get('con3d_loss', 0) \
                            + loss_dict.get('con2d_loss', 0)

        return loss_dict


if __name__ == '__main__':
    """Test the model
    """
    from LSG.main import setup
    from options.cfg_options import CFGOptions
    args = CFGOptions().parse()
    args.config_file = 'E:\LiteSpiralGCN\LSG\configs\LiteSpiralGCN.yml'
    cfg = setup(args)

    model = LiteSpiralGCN(cfg)
    model_out = model(torch.zeros(2,3, 128, 128))
    print(model_out['verts'].size())
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters: ", total_params / 1000000)

    # 使用 thop 计算参数量和 FLOPs
    flops, params = profile(model, inputs=(torch.zeros(1,3, 128, 128),))
    print("Total parameters: {:.2f} million".format(params / 1_000_000))
    print("Total FLOPs: {:.2f} million".format(flops / 1_000_000))