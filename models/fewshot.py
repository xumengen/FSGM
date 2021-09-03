# -*- encoding: utf-8 -*-
'''
@File    :   fewshot.py
@Author  :   Mengen Xu
@Version :   1.0
@Contact :   mengen0120@gmail.com
'''


"""
Fewshot Semantic Segmentation
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg import Encoder
import segmentation_models_pytorch as smp


class FewShotSeg(nn.Module):
    """
    Fewshot Segmentation model

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """
    def __init__(self, in_channels=3, pretrained_path=None, config=None):
        super().__init__()
        self.pretrained_path = pretrained_path

        # Encoder
        if config['encoder'] == 'VGG':
            self.encoder = nn.Sequential(OrderedDict([
                 ('backbone', Encoder(in_channels, self.pretrained_path)),]))

        elif config['encoder'] == 'FPN':
            self.encoder = smp.FPN(
                encoder_name="resnet101",       
                encoder_weights="imagenet",     
                in_channels=3,                  
                classes=config['output_feature_length'],                     
            )
        elif config['encoder'] == 'FPNnopretrain':
            self.encoder = smp.FPN(
                encoder_name="resnet101",        
                # encoder_weights="imagenet",    
                in_channels=3,                  
                classes=config['output_feature_length'],                     
            )


    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0),], dim=0)
        img_fts = self.encoder(imgs_concat)
        fts_size = img_fts.shape[-2:]

        supp_fts = img_fts[:n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * batch_size:].view(
            n_queries, batch_size, -1, *fts_size)   # N x B x C x H' x W'
        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H x W
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Wa x Sh x B x H x W

        for epi in range(batch_size):
            supp_fg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             fore_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]
            supp_bg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             back_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]

        return supp_fg_fts, supp_bg_fts, qry_fts


    def getFeatures(self, fts, mask):

        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')  # 1 * C * H * W
        positive_mask = torch.nonzero(mask, as_tuple=True)

        return fts[:, :, positive_mask[-2], positive_mask[-1]]
