# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import timm
from einops import rearrange
import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from .swin_transformer_ROPE import SwinTransformerSys_ROPE
from .swin_transformer_DCT import SwinTransformerSys_DCT
logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=1000, zero_head=False, vis=False,in_chans=1,
                 depths=[2,2,6,2], embed_dim=96, patch_size=4):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.depths = depths
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.swin_unet = SwinTransformerSys(img_size=config.img_size,
                                patch_size=self.patch_size,
                                in_chans=in_chans,
                                num_classes=self.num_classes,
                                embed_dim=self.embed_dim,
                                depths=self.depths,
                                num_heads=config.num_heads,
                                window_size=config.window_size,
                                mlp_ratio=config.mlp_ratio,
                                qkv_bias=config.qkv_bias,
                                qk_scale=config.qk_scale,
                                drop_rate=config.drop_rate,
                                drop_path_rate=config.drop_path_rate,
                                ape=config.ape,
                                patch_norm=config.patch_norm,
                                use_checkpoint=config.use_checkpoint)

    def forward(self, x):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")


class SwinUnet_ROPE(nn.Module):
    def __init__(self, config, img_size=224, num_classes=1000, zero_head=False, vis=False, in_chans=1,
                 depths=[2, 2, 6, 2], embed_dim=96, patch_size=4):
        super(SwinUnet_ROPE, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.depths = depths
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.swin_unet_ROPE = SwinTransformerSys_ROPE(img_size=config.img_size,
                                            patch_size=self.patch_size,
                                            in_chans=in_chans,
                                            num_classes=self.num_classes,
                                            embed_dim=self.embed_dim,
                                            depths=self.depths,
                                            num_heads=config.num_heads,
                                            window_size=config.window_size,
                                            mlp_ratio=config.mlp_ratio,
                                            qkv_bias=config.qkv_bias,
                                            qk_scale=config.qk_scale,
                                            drop_rate=config.drop_rate,
                                            drop_path_rate=config.drop_path_rate,
                                            ape=config.ape,
                                            patch_norm=config.patch_norm,
                                            use_checkpoint=config.use_checkpoint)

    def forward(self, x):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,3,1,1)
        logits = self.swin_unet_ROPE(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")


class SwinUnet_DCT(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False,in_chans=1,
                 depths=[2,2,6,2], embed_dim=96, patch_size= 4):
        super(SwinUnet_DCT, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.depths = depths
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.swin_unet = SwinTransformerSys_DCT(img_size=config.img_size,
                                patch_size=self.patch_size,
                                in_chans=in_chans,
                                num_classes=self.num_classes,
                                embed_dim=self.embed_dim,
                                depths=self.depths,
                                num_heads=config.num_heads,
                                window_size=config.window_size,
                                mlp_ratio=config.mlp_ratio,
                                qkv_bias=config.qkv_bias,
                                qk_scale=config.qk_scale,
                                drop_rate=config.drop_rate,
                                drop_path_rate=config.drop_path_rate,
                                ape=config.ape,
                                patch_norm=config.patch_norm,
                                use_checkpoint=config.use_checkpoint)

    def forward(self, x):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
