# @Time : 2023/5/30
# @Author : Z.chang
# @FileName: fewshot.py
# @Software: PyCharm
# @Description：Few-shot

from collections import OrderedDict
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.vgg import Encoder
from models import resnet_50_101
from models.vit_model import VisionTransformer
from functools import partial
import math
# from pytorch_pretrained_vit import ViT
from util import utils
import numpy as np
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from models.cross_attention import cross_att


class FewShotSeg(nn.Module):
    """
       Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """

    def __init__(self, in_channels=3, pretrained_path=None, cfg=None, depth=12, act_layer=None, norm_layer=None):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}
        # # # Encoder
        self.encoder = nn.Sequential(OrderedDict([
            ('backbone', Encoder(in_channels, self.pretrained_path)), ]))
        # self.encoder = nn.Sequential(OrderedDict([
        #     ('backbone', resnet_50_101.resnet50(pretrained=True)), ]))

        self.proj = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=1)
        self.channel = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.vit_model = VisionTransformer(img_size=448,
                                           patch_size=32,
                                           in_c=512,
                                           embed_dim=512,
                                           # embed_dim=1024,
                                           depth=12,
                                           num_heads=16,
                                           # distilled=True,
                                           representation_size=None,
                                           num_classes=0)
        self.cross_att = cross_att(56)

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
        ###### Extract and map features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        img_fts_proj_out = self.encoder(imgs_concat)  # 2 1024 56 56
        # img_fts_proj_out = self.proj(img_fts_resnet_out)  # 2 512 56 56
        fts_size = img_fts_proj_out.shape[-2:]  # 最后输出的维度
        supp_fts_proj_out = img_fts_proj_out[:n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size)  # [1 1 1 512 56 56] # support_Way x Shot x B x C x H' x W'
        qry_fts_proj_out = img_fts_proj_out[n_ways * n_shots * batch_size:].view(
            n_queries, batch_size, -1, *fts_size)  # query_way x B x C x H' x W' [1 1 512 56 56]

        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Way x Shot x B x H x W [1, 1, 1, 448, 448]
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Way x Shot x B x H x W [1, 1, 1, 448, 448]
        ###### Compute loss ######

        # query_mask ->support mask
        query_outputs_foreground = []
        support_outputs_foreground = []
        query_outputs_background = []
        support_outputs_background = []
        qry_fts_bg = qry_fts_proj_out.squeeze()  # 512 56 56
        # qry_fts_fg = qry_fts_proj_out.squeeze()
        for epi in range(batch_size):
            supp_fg_fts = []
            for way in range(n_ways):
                shot_list = []
                for shot in range(n_shots):
                    supp_fts_fg_proj_out_way_shot_epi = supp_fts_proj_out[way, shot, [epi]]
                    fore_mask_way_shot_epi = fore_mask[way, shot, [epi]]
                    fg_fts = self.getFeatures(supp_fts_fg_proj_out_way_shot_epi, fore_mask_way_shot_epi)
                    key_fg_sup = query_fg_sup = value_fg_sup = F.interpolate(fg_fts[..., None, None], size=56,
                                                                             mode='bilinear').squeeze()  # [512 56 56]
                    "---------------支持前景自注意力------------"
                    self_fg_supp_att = self.cross_att(key_fg_sup, query_fg_sup, value_fg_sup)  # [512 56 56]
                    "---------------查询前景自注意力------------"
                    key_bg_que = query_bg_que = value_bg_que = qry_fts_bg  # [512 56 56]
                    self_bg_que_att = self.cross_att(key_bg_que, query_bg_que, value_bg_que)  # [512 56 56]
                    "------------支持前景、查询前景特征融合------"
                    fusion_fg_bg = self_bg_que_att + self_fg_supp_att
                    "-----------支持前景、查询前景交叉注意力（自注意力的交叉注意力融合）-----"
                    # query_fg_que = value_fg_que = qry_fts_fg  # [512 56 56]
                    cross_fg_att = self.cross_att(self_fg_supp_att, self_bg_que_att, self_fg_supp_att)  # [512 56 56]
                    "---------------自注意力和交叉注意力特征融合------------------"
                    # cross_self_att_fusion = fusion_fg_bg + cross_fg_att  # [512 56 56]
                    final_fusion_fts = self.getFeatures(cross_fg_att.unsqueeze(dim=0),
                                                        fore_mask_way_shot_epi)  # [1 512]
                    shot_list.append(final_fusion_fts)
                supp_fg_fts.append(shot_list)
            "--------------------------------The foreground regions of support images end----------------------------------"
            supp_bg_fts = []
            for way in range(n_ways):
                shot_list = []
                for shot in range(n_shots):
                    supp_fts_bg_proj_out_way_shot_epi = supp_fts_proj_out[way, shot, [epi]]
                    bg_mask_way_shot_epi = back_mask[way, shot, [epi]]
                    bg_fts = self.getFeatures(supp_fts_bg_proj_out_way_shot_epi, bg_mask_way_shot_epi)
                    up_fts = F.interpolate(bg_fts[..., None, None], size=back_mask.shape[-2:], mode='bilinear')
                    vit_fts = self.vit_model(up_fts)
                    handle_fts = self.handle_vit(vit_fts.reshape((1, 14, 14, 512)).permute([0, 3, 2, 1]),
                                                 bg_mask_way_shot_epi)
                    shot_list.append(handle_fts)
                supp_bg_fts.append(shot_list)
            "--------------------------------The background regions of support images end-------------------------------"
            fg_prototypes, bg_prototype = self.getPrototype(supp_fg_fts, supp_bg_fts)

            foreground_prototypes = [bg_prototype, ] + fg_prototypes  
            dist1 = [self.calDist(qry_fts_proj_out[:, epi], prototype) for prototype in foreground_prototypes]
            query_foreground_pred = torch.stack(dist1, dim=1)  #
            query_outputs_foreground.append(F.interpolate(query_foreground_pred, size=img_size, mode='bilinear'))
            "-------------------------------The prediction of objects in the query image end-------------------------"
            for way in range(n_ways):
                for shot in range(n_shots):
                    supp_fts_proj_out = supp_fts_proj_out.squeeze()  # [5shot 512 56 56]
                    shot_feature_01 = supp_fts_proj_out[shot]
                    shot_feature_02 = shot_feature_01.unsqueeze(dim=0)
                    shot_feature = shot_feature_02.unsqueeze(dim=0)
                    dist2 = [self.calDist(shot_feature[:, epi], prototype) for prototype in foreground_prototypes]
                    support_foreground_pred = torch.stack(dist2, dim=1)  #
                    support_outputs_foreground.append(
                        F.interpolate(support_foreground_pred, size=img_size, mode='bilinear'))
            "-------------------------------The prediction of objects in the support image end-------------------------"
            query_bg_fts = []
            for way in range(n_ways):
                shot_list = []
                for shot in range(n_shots):
                    pooling_layer = nn.AdaptiveAvgPool2d((1, 1))
                    fts = pooling_layer(qry_fts_bg.unsqueeze(dim=0))
                    mean_fts = fts.view(-1, 512)
                    # mean_fts = torch.mean(qry_fts_bg, dim=(1, 2)).unsqueeze(dim=0) #
                    shot_list.append(mean_fts)  # [1 512]
                query_bg_fts.append(shot_list)
            "--------------------------------The feature extraction of query background regions end----------------------------------"
            background_que_prototype, background_sup_prototypes = self.getPrototype(query_bg_fts, supp_bg_fts)
            ###### Compute the distance ######
            background_prototypes = [background_sup_prototypes, ] + background_que_prototype
            dist3 = [self.calDist(qry_fts_proj_out[:, epi], prototype) for prototype in background_prototypes]
            query_background_pred = torch.stack(dist3, dim=1)  #
            query_outputs_background.append(F.interpolate(query_background_pred, size=img_size, mode='bilinear'))
            "-------------------------------The prediction of background in the support image start-------------------------"
            for way in range(n_ways):
                for shot in range(n_shots):
                    supp_fts_proj_out = supp_fts_proj_out.squeeze()  # [5 512 56 56]
                    shot_feature_01 = supp_fts_proj_out[shot]
                    shot_feature_02 = shot_feature_01.unsqueeze(dim=0)
                    shot_feature = shot_feature_02.unsqueeze(dim=0)
                    dist4 = [self.calDist(shot_feature[:, epi], prototype) for prototype in background_prototypes]
                    support_background_pred = torch.stack(dist4, dim=1)  #
                    support_outputs_background.append(
                        F.interpolate(support_background_pred, size=img_size, mode='bilinear'))
            "-------------------------------训练-超参数设置-------------------------"
            ###### Prototype alignment loss ######
            # 测试阶段 self.training为Flase 说明测试阶段没有执行if
            # if self.config['align'] and self.training:
            # flag = True  # 定义一个flag  True则执行CG, False 则不执行CG
            # if self.config['align'] and flag:
            #     query_loss_foreground = self.alignLoss(qry_fts_proj_out[:, epi], query_foreground_pred, supp_fts_proj_out[:, :, epi], fore_mask[:, :, epi], back_mask[:, :, epi])
            #     query_loss_background = self.alignLoss(qry_fts_proj_out[:, epi], query_background_pred, supp_fts_proj_out[:, :, epi], fore_mask[:, :, epi], back_mask[:, :, epi])
            #     support_loss_foreground = self.alignLoss(supp_fts_proj_out[:, :, epi], support_foreground_pred, supp_fts_proj_out[:, :, epi], fore_mask[:, :, epi], back_mask[:, :, epi])
            #     support_loss_background = self.alignLoss(supp_fts_proj_out[:, :, epi], support_background_pred, supp_fts_proj_out[:, :, epi], fore_mask[:, :, epi], back_mask[:, :, epi])
            #
            #
            #
            #
            #     query_ls1 += query_loss_foreground
            #     query_ls2 += query_loss_background
            #     align_loss_total = query_ls1 + query_ls2

        query_output_foreground = torch.stack(query_outputs_foreground, dim=1)  # N x B x (1 + Wa) x H x W
        que_output_foreground = query_output_foreground.view(-1, *query_output_foreground.shape[2:])
        query_output_background = torch.stack(query_outputs_background, dim=1)  # N x B x (1 + Wa) x H x W
        que_output_background = query_output_background.view(-1, *query_output_background.shape[2:])

        support_output_foreground = torch.stack(support_outputs_foreground, dim=1)  # N x B x (1 + Wa) x H x W
        sup_output_foreground = support_output_foreground.view(-1, *support_output_foreground.shape[2:])
        support_output_background = torch.stack(support_outputs_background, dim=1)  # N x B x (1 + Wa) x H x W
        sup_output_background = support_output_background.view(-1, *support_output_background.shape[2:])

        return que_output_foreground, que_output_background, sup_output_foreground, sup_output_background  # [1 2 448 448] [shot 2 448 448]
    # def self_attention(query, key, value, mask=None, dropout=None):
    #
    #     d_k = query.size(-1)
    #     # (nbatch, h, seq_len, d_k) @ (nbatch, h, d_k, seq_len) => (nbatch, h, seq_len, seq_len)
    #     scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    #     if mask is not None:
    #         scores = scores.masked_fill(mask == 0, -1e9)
    #     p_attn = F.softmax(scores, dim=-1)
    #     if dropout:
    #         p_attn = dropout(p_attn)
    #     # (nbatch, h, seq_len, seq_len) * (nbatch, h, seq_len, d_k) = > (nbatch, h, seq_len, d_k)
    #     return torch.matmul(p_attn, value), p_attn

    def calDist(self, query_cnn_out, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        dist = F.cosine_similarity(query_cnn_out, prototype[..., None, None], dim=1) * scaler
        return dist

    def getFeatures(self, fts, mask):
        fts = F.interpolate(fts, size=mask.shape[-2:],
                            mode='bilinear')  # 默认nearest, linear(3D-only), bilinear(4D-only), trilinear(5D-only)

        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)  # 1 x C
        return masked_fts  # [1 512]

    #  @GL 针对vit前后 进行mask 以及sum
    def handle_vit(self, fts, mask):
        # fts = F.interpolate(fts, size=mask.shape[-2:],
        #                     mode='bilinear')  # 默认nearest, linear(3D-only), bilinear(4D-only), trilinear(5D-only)
        # if is_fore_vit:  # 送入vit前mask
        #     masked_fts = fts * mask[None, ...]
        # else:  # vit 出来进行sum
        #     masked_fts = torch.sum(fts, dim=(2, 3)) \
        #                  / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)  # 1 x C
        masked_fts = torch.sum(fts, dim=(2, 3)) \
                     / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)
        return masked_fts

    # @czb ################通过平均前景和背景特征获得原型###############
    def getPrototype(self, fg_fts, bg_fts):  # param: 1*512*56*56,  1*512*14*14
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])  # 1， 5
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        return fg_prototypes, bg_prototype  # return [tensor], tensor   一个list 一个tensor?

    ############## # @CZB过渡段学习CCG(Query->Support)##################
    #####@
    # def alignLoss(self, query_vgg_out, pred, support_resnet_out, support_fore_mask, support_back_mask):
    #     """
    #     Compute the loss for the prototype alignment branch
    #
    #     Args:
    #         query_resnet_out: embedding features for query images
    #             expect shape: N x C x H' x W'
    #         pred: predicted segmentation score
    #             expect shape: N x (1 + Way) x H x W
    #         support_resnet_out: embedding features for support images
    #             expect shape: Way x Shot x C x H' x W'
    #         support_fore_mask: foreground masks for support images
    #             expect shape: way x shot x H x W
    #         support_back_mask: background masks for support images
    #             expect shape: way x shot x H x W
    #     """
    #     n_ways, n_shots = len(support_fore_mask), len(support_fore_mask[0])
    #     # Mask and get query prototype
    #     pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
    #     binary_masks = [pred_mask == i for i in range(1 + n_ways)]  # 前景+1个背景
    #     skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]  # 没懂
    #     ##########@czb query-mask########
    #     pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Way) x 1 x H' x W'
    #     query_prototypes = torch.sum(query_vgg_out.unsqueeze(1) * pred_mask, dim=(0, 3, 4))
    #     ###########获取query的原型###########
    #     query_prototypes = query_prototypes / (pred_mask.sum((0, 3, 4)) + 1e-5)  # (1 + Way) x C
    #     # Compute the support loss
    #     loss = 0
    #     for way in range(n_ways):
    #         if way in skip_ways:
    #             continue
    #         # Get the query prototypes
    #         prototypes = [query_prototypes[[0]], query_prototypes[[way + 1]]]
    #         for shot in range(n_shots):
    #             img_fts = support_resnet_out[way, [shot]]
    #             supp_dist = [self.calDist(img_fts, prototype) for prototype in prototypes]
    #             supp_pred = torch.stack(supp_dist, dim=1)
    #             supp_pred = F.interpolate(supp_pred, size=support_fore_mask.shape[-2:],
    #                                       mode='bilinear')
    #             # Construct the support Ground-Truth segmentation
    #             supp_label = torch.full_like(support_fore_mask[way, shot], 255,
    #                                          device=img_fts.device).long()
    #             supp_label[support_fore_mask[way, shot] == 1] = 1
    #             supp_label[support_back_mask[way, shot] == 1] = 0
    #             # Compute Loss
    #             loss = loss + F.cross_entropy(
    #                 supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways
    #     return loss
