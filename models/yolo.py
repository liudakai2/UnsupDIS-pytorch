"""YOLOv5-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path
import numpy as np
import time
from collections import Iterable

sys.path.append(Path(__file__).parent.parent.absolute().__str__())  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.transforms.functional import perspective  # crash on cuda tensors. stupid.
from models.common import *
from models.experimental import *
from utils.general import make_divisible, check_file, set_logging
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, check_anomaly
from utils.stitching import DLTSolver, STN

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter
    
    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)
    
    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                
                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))
        
        return x if self.training else (torch.cat(z, 1), x)
    
    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
    

class HEstimator(nn.Module):
    def __init__(self, input_size=128, strides=(2,4,8), keep_prob=0.5, ch=()):
        super(HEstimator, self).__init__()
        self.ch = ch  # channels for multiple feature maps, e.g., [48, 96, 192] for yolov5m
        self.stride = torch.tensor([4, 32])  # fake
        self.input_size = input_size
        self.strides = strides
        self.search_ranges = [16, 8, 4]
        self.patch_sizes = [32., 64., 128.]
        # shape[2, 2, 3, 3]
        self.aux_matrices = torch.stack([self.gen_aux_mat(patch_size) for patch_size in self.patch_sizes])
        self.DLT_solver = DLTSolver()
        
        m = []
        s = input_size // (128 // 8)
        k, p = s, 0
        for i, x in enumerate(ch[::-1]):  # manually calculate the channels
            # ch1 = (self.search_ranges[i] * 2 + 1) ** 2
            ch1 = x * 2
            ch_conv = 512 // (2 ** i)
            # ch_flat = (input_size // strides[-(i+1)] // s) ** 2 * ch_conv
            ch_flat = (input_size // strides[-1] // s) ** 2 * ch_conv
            # ch_flat = ch_conv * (s ** 2)
            ch_fc = 512 // (2 ** i)
            # print(x, ch1, ch_conv, ch_flat, ch_fc)
            m.append(
                nn.Sequential(
                    Conv(ch1, ch_conv, k=3, s=1),
                    Conv(ch_conv, ch_conv, k=3, s=2 if i >= 2 else 1),  # stage 2
                    Conv(ch_conv, ch_conv, k=3, s=2 if i >= 1 else 1),  # stage 2 & 3
                    DWConv(ch_conv, ch_conv, k=k, s=s, p=p),
                    # nn.AvgPool2d(k, s, p),
                    # nn.AdaptiveAvgPool2d((s, s)),
                    nn.Flatten(),
                    nn.Linear(ch_flat, ch_fc),
                    nn.SiLU(),
                    # nn.Dropout(keep_prob),
                    nn.Linear(ch_fc, 8, bias=False)
                )
            )
        self.m = nn.ModuleList(m)
    
    def forward(self, feature1, feature2, image2, mask2):
        bs = image2.size(0)
        assert len(self.search_ranges) == len(feature1) == len(feature2)
        device, dtype = image2.device, image2.dtype
        if self.aux_matrices.device != device:
            self.aux_matrices = self.aux_matrices.to(device)
        if self.aux_matrices.dtype != dtype:
            self.aux_matrices = self.aux_matrices.type(dtype)
            
        vertices_offsets = []
        for i, search_range in enumerate(self.search_ranges):
            # global_correlation is either time-consuming or memory-consuming, and even leads to divergence
            # concatenation seems to be capable enough for estimation
            # x = self.cost_volume(feature1[-(i+1)], feature2[-(i+1)], search_range)
            x = torch.cat((feature1[-(i+1)], feature2[-(i+1)]), dim=1)
            
            off = self.m[i](x).unsqueeze(-1)  # [bs, 8, 1], for matrix multiplication
            assert torch.isnan(off).sum() == 0
            
            # off, overflow = self.clip_offset(off, vertices_offsets, phase=i)
            vertices_offsets.append(off)
            
            if i == len(self.search_ranges) - 1:
                break
            
            # H = self.DLT_solver.solve(sum(vertices_offsets) / (2 ** (2 - i)), self.patch_sizes[i])  # 2x up-scale
            # M, M_inv = torch.chunk(self.aux_matrices[i], 2, dim=0)
            # 4x down-scale for numerical stability
            H = self.DLT_solver.solve(sum(vertices_offsets) / 4., self.patch_sizes[0])
            M, M_inv = torch.chunk(self.aux_matrices[0], 2, dim=0)

            H = torch.bmm(torch.bmm(M_inv.expand(bs, -1, -1), H), M.expand(bs, -1, -1))

            feature2[-(i + 2)] = STN(feature2[-(i + 2)], H, vertices_offsets)

        warped_imgs, warped_msks = [], []
        patch_level = 0
        M, M_inv = torch.chunk(self.aux_matrices[patch_level], 2, dim=0)
        img_with_msk = torch.cat((image2, mask2), dim=1)
        for i in range(len(vertices_offsets)):
            H_inv = self.DLT_solver.solve(sum(vertices_offsets[:i+1]) / (2 ** (2 - patch_level)), self.patch_sizes[patch_level])
            H = torch.bmm(torch.bmm(M_inv.expand(bs, -1, -1), H_inv), M.expand(bs, -1, -1))
            warped_img, warped_msk = STN(img_with_msk, H, vertices_offsets[:i+1]).split([3, 1], dim=1)
            warped_imgs.append(warped_img)
            warped_msks.append(warped_msk)
        
        # the relationship (or definition) between `H` and `H_inv` is confusing
        # H = torch.linalg.inv(H_inv.detach())
        # H /= H[:, -1, -1]  # same as the results from cv2.getPerspectiveTransform(org, dst)
        
        return sum(vertices_offsets), warped_imgs, warped_msks

    @staticmethod
    def cost_volume(x1, x2, search_range, norm=True, fast=True):
        if norm:
            x1 = F.normalize(x1, p=2, dim=1)
            x2 = F.normalize(x2, p=2, dim=1)
        bs, c, h, w = x1.shape
        padded_x2 = F.pad(x2, [search_range] * 4)  # [b,c,h,w] -> [b,c,h+sr*2,w+sr*2]
        max_offset = search_range * 2 + 1

        if fast:
            # faster(*2) but cost higher(*n) GPU memory
            patches = F.unfold(padded_x2, (max_offset, max_offset)).reshape(bs, c, max_offset ** 2, h, w)
            cost_vol = (x1.unsqueeze(2) * patches).mean(dim=1, keepdim=False)
        else:
            # slower but save memory
            cost_vol = []
            for j in range(0, max_offset):
                for i in range(0, max_offset):
                    x2_slice = padded_x2[:, :, j:j + h, i:i + w]
                    cost = torch.mean(x1 * x2_slice, dim=1, keepdim=True)
                    cost_vol.append(cost)
            cost_vol = torch.cat(cost_vol, dim=1)
        
        cost_vol = F.leaky_relu(cost_vol, 0.1)
        
        return cost_vol
    
    @staticmethod
    def gen_aux_mat(patch_size):
        M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                      [0., patch_size / 2.0, patch_size / 2.0],
                      [0., 0., 1.]]).astype(np.float32)
        M_inv = np.linalg.inv(M)
        return torch.from_numpy(np.stack((M, M_inv)))  # [2, 3, 3]


class HEstimatorOrigin(HEstimator):
    def __init__(self, input_size=128, strides=(2,4,8), keep_prob=0.5, ch=()):
        super().__init__()
        self.ch = ch  # channels for multiple feature maps, e.g., [48, 96, 192] for yolov5m
        self.stride = torch.tensor([4, 32])  # fake
        self.input_size = input_size
        self.strides = strides
        self.search_ranges = [16, 8, 4]
        self.patch_sizes = [32., 64., 128.]
        # shape[2, 2, 3, 3]
        self.aux_matrices = torch.stack([self.gen_aux_mat(patch_size) for patch_size in self.patch_sizes])
        self.DLT_solver = DLTSolver()
        
        m = []
        for i, x in enumerate(ch[::-1]):  # manually calculate the channels
            ch1 = (self.search_ranges[i] * 2 + 1) ** 2
            ch_conv = 512 // (2 ** i)
            ch_flat = (input_size // strides[-1]) ** 2 * ch_conv
            ch_fc = 1024 // (2 ** i)
            # print(x, ch1, ch_conv, ch_flat, ch_fc)
            m.append(
                nn.Sequential(
                    ConvWoBN(ch1, ch_conv, k=3, s=1),
                    ConvWoBN(ch_conv, ch_conv, k=3, s=2 if i >= 2 else 1),  # stage 2
                    ConvWoBN(ch_conv, ch_conv, k=3, s=2 if i >= 1 else 1),  # stage 2 & 3
                    nn.Flatten(),
                    nn.Linear(ch_flat, ch_fc),
                    nn.ReLU(),
                    nn.Dropout(keep_prob),
                    nn.Linear(ch_fc, 8, bias=False)
                )
            )
        self.m = nn.ModuleList(m)
    
    def forward(self, feature1, feature2, image2, mask2):
        bs = image2.size(0)
        assert len(self.search_ranges) == len(feature1) == len(feature2)
        device, dtype = image2.device, image2.dtype
        if self.aux_matrices.device != device:
            self.aux_matrices = self.aux_matrices.to(device)
        if self.aux_matrices.dtype != dtype:
            self.aux_matrices = self.aux_matrices.type(dtype)
            
        vertices_offsets = []
        for i, search_range in enumerate(self.search_ranges):
            # global_correlation is either time-consuming or memory-consuming, and even leads to divergence
            # concatenation seems to be capable enough for estimation
            x = self.cost_volume(feature1[-(i+1)], feature2[-(i+1)], search_range)
            # x = torch.cat((feature1[-(i+1)], feature2[-(i+1)]), dim=1)
            
            off = self.m[i](x).unsqueeze(-1)  # [bs, 8, 1], for matrix multiplication
            assert torch.isnan(off).sum() == 0
            
            # off, overflow = self.clip_offset(off, vertices_offsets, phase=i)
            vertices_offsets.append(off)
            
            if i == len(self.search_ranges) - 1:
                break
            
            # H = self.DLT_solver.solve(sum(vertices_offsets) / (2 ** (2 - i)), self.patch_sizes[i])  # 2x up-scale
            # M, M_inv = torch.chunk(self.aux_matrices[i], 2, dim=0)
            # 4x down-scale for numerical stability
            H = self.DLT_solver.solve(sum(vertices_offsets) / 4., self.patch_sizes[0])
            M, M_inv = torch.chunk(self.aux_matrices[0], 2, dim=0)

            H = torch.bmm(torch.bmm(M_inv.expand(bs, -1, -1), H), M.expand(bs, -1, -1))

            feature2[-(i + 2)] = STN(feature2[-(i + 2)], H, vertices_offsets)

        warped_imgs, warped_msks = [], []
        patch_level = 0
        M, M_inv = torch.chunk(self.aux_matrices[patch_level], 2, dim=0)
        img_with_msk = torch.cat((image2, mask2), dim=1)
        for i in range(len(vertices_offsets)):
            H_inv = self.DLT_solver.solve(sum(vertices_offsets[:i+1]) / (2 ** (2 - patch_level)), self.patch_sizes[patch_level])
            H = torch.bmm(torch.bmm(M_inv.expand(bs, -1, -1), H_inv), M.expand(bs, -1, -1))
            warped_img, warped_msk = STN(img_with_msk, H, vertices_offsets[:i+1]).split([3, 1], dim=1)
            warped_imgs.append(warped_img)
            warped_msks.append(warped_msk)
        
        # the relationship (or definition) between `H` and `H_inv` is confusing
        # H = torch.linalg.inv(H_inv.detach())
        # H /= H[:, -1, -1]  # same as the results from cv2.getPerspectiveTransform(org, dst)
        
        return sum(vertices_offsets), warped_imgs, warped_msks


class HEstimatorVariant(nn.Module):
    def __init__(self, input_size=128, strides=(2,4,8), keep_prob=0.5, ch=()):
        super(HEstimatorVariant, self).__init__()
        self.ch = ch  # channels for multiple feature maps, e.g., [48, 96, 192] for yolov5m
        self.stride = torch.tensor([4, 32])  # fake
        self.input_size = input_size
        self.strides = strides
        self.search_ranges = [16, 8, 4]
        self.patch_sizes = [32., 64., 128.]
        # shape[2, 2, 3, 3]
        self.aux_matrices = torch.stack([self.gen_aux_mat(patch_size) for patch_size in self.patch_sizes])
        self.DLT_solver = DLTSolver()
        
        m = []
        s = input_size // (128 // 8)
        k, p = s, 0
        for i, x in enumerate(ch[::-1]):  # manually calculate the channels
            # ch1 = (self.search_ranges[i] * 2 + 1) ** 2
            ch1 = x * 2
            ch_conv = 512 // (2 ** i)
            # ch_flat = (input_size // strides[-(i+1)] // s) ** 2 * ch_conv
            ch_flat = (input_size // strides[-1] // s) ** 2 * ch_conv
            # ch_flat = ch_conv * (s ** 2)
            ch_fc = 512 // (2 ** i)
            # print(x, ch1, ch_conv, ch_flat, ch_fc)
            m.append(
                nn.Sequential(
                    Conv(ch1, ch_conv, k=3, s=1, norm='HIN'),
                    Conv(ch_conv, ch_conv, k=3, s=2 if i >= 2 else 1, norm='HIN'),  # stage 2
                    Conv(ch_conv, ch_conv, k=3, s=2 if i >= 1 else 1, norm='HIN'),  # stage 2 & 3
                    DWConv(ch_conv, ch_conv, k=k, s=s, p=p),
                    # nn.AvgPool2d(k, s, p),
                    # nn.AdaptiveAvgPool2d((s, s)),
                    nn.Flatten(),
                    nn.Linear(ch_flat, ch_fc),
                    nn.SiLU(),
                    # nn.Dropout(keep_prob),
                    nn.Linear(ch_fc, 8, bias=False)
                )
            )
        self.m = nn.ModuleList(m)
    
    def forward(self, feature1, feature2, image2, mask2):
        bs = image2.size(0)
        assert len(self.search_ranges) == len(feature1) == len(feature2)
        device, dtype = image2.device, image2.dtype
        if self.aux_matrices.device != device:
            self.aux_matrices = self.aux_matrices.to(device)
        if self.aux_matrices.dtype != dtype:
            self.aux_matrices = self.aux_matrices.type(dtype)
            
        vertices_offsets = []
        for i, search_range in enumerate(self.search_ranges):
            # global_correlation is either time-consuming or memory-consuming, and even leads to divergence
            # concatenation seems to be capable enough for estimation
            # x = self.cost_volume(feature1[-(i+1)], feature2[-(i+1)], search_range)
            x = torch.cat((feature1[-(i+1)], feature2[-(i+1)]), dim=1)
            
            off = self.m[i](x).unsqueeze(-1)  # [bs, 8, 1], for matrix multiplication
            assert torch.isnan(off).sum() == 0
            
            # off, overflow = self.clip_offset(off, vertices_offsets, phase=i)
            vertices_offsets.append(off)
            
            if i == len(self.search_ranges) - 1:
                break
            
            # H = self.DLT_solver.solve(sum(vertices_offsets) / (2 ** (2 - i)), self.patch_sizes[i])  # 2x up-scale
            # M, M_inv = torch.chunk(self.aux_matrices[i], 2, dim=0)
            # 4x down-scale for numerical stability
            H = self.DLT_solver.solve(sum(vertices_offsets) / 4., self.patch_sizes[0])
            M, M_inv = torch.chunk(self.aux_matrices[0], 2, dim=0)

            H = torch.bmm(torch.bmm(M_inv.expand(bs, -1, -1), H), M.expand(bs, -1, -1))

            feature2[-(i + 2)] = STN(feature2[-(i + 2)], H, vertices_offsets)

        warped_imgs, warped_msks = [], []
        patch_level = 0
        M, M_inv = torch.chunk(self.aux_matrices[patch_level], 2, dim=0)
        img_with_msk = torch.cat((image2, mask2), dim=1)
        for i in range(len(vertices_offsets)):
            H_inv = self.DLT_solver.solve(sum(vertices_offsets[:i+1]) / (2 ** (2 - patch_level)), self.patch_sizes[patch_level])
            H = torch.bmm(torch.bmm(M_inv.expand(bs, -1, -1), H_inv), M.expand(bs, -1, -1))
            warped_img, warped_msk = STN(img_with_msk, H, vertices_offsets[:i+1]).split([3, 1], dim=1)
            warped_imgs.append(warped_img)
            warped_msks.append(warped_msk)
        
        # the relationship (or definition) between `H` and `H_inv` is confusing
        # H = torch.linalg.inv(H_inv.detach())
        # H /= H[:, -1, -1]  # same as the results from cv2.getPerspectiveTransform(org, dst)
        
        return sum(vertices_offsets), warped_imgs, warped_msks

    @staticmethod
    def cost_volume(x1, x2, search_range, norm=True, fast=True):
        if norm:
            x1 = F.normalize(x1, p=2, dim=1)
            x2 = F.normalize(x2, p=2, dim=1)
        bs, c, h, w = x1.shape
        padded_x2 = F.pad(x2, [search_range] * 4)  # [b,c,h,w] -> [b,c,h+sr*2,w+sr*2]
        max_offset = search_range * 2 + 1

        if fast:
            # faster(*2) but cost higher(*n) GPU memory
            patches = F.unfold(padded_x2, (max_offset, max_offset)).reshape(bs, c, max_offset ** 2, h, w)
            cost_vol = (x1.unsqueeze(2) * patches).mean(dim=1, keepdim=False)
        else:
            # slower but save memory
            cost_vol = []
            for j in range(0, max_offset):
                for i in range(0, max_offset):
                    x2_slice = padded_x2[:, :, j:j + h, i:i + w]
                    cost = torch.mean(x1 * x2_slice, dim=1, keepdim=True)
                    cost_vol.append(cost)
            cost_vol = torch.cat(cost_vol, dim=1)
        
        cost_vol = F.leaky_relu(cost_vol, 0.1)
        
        return cost_vol
    
    @staticmethod
    def gen_aux_mat(patch_size):
        M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                      [0., patch_size / 2.0, patch_size / 2.0],
                      [0., 0., 1.]]).astype(np.float32)
        M_inv = np.linalg.inv(M)
        return torch.from_numpy(np.stack((M, M_inv)))  # [2, 3, 3]

class Reconstructor(nn.Module):
    def __init__(self, ch=()):
        super(Reconstructor, self).__init__()
        ch_lr = ch[0]
        self.m_lr = nn.Sequential(
            # nn.Conv2d(ch_lr, ch_lr, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.ReLU(inplace=True),
            nn.Conv2d(ch_lr, 3, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.m_hr = nn.Sequential(
            nn.Conv2d(3 * 3, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            C3(64, 64, 3),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.stride = torch.tensor([4, 32])  # fake
    
    def forward(self, x):
        out_lr, in_hr = x
        # low resolution
        out_lr = self.m_lr(out_lr).sigmoid_()
        # super resolution
        out_lr_sr = F.interpolate(out_lr, mode='bilinear', size=in_hr.shape[2:], align_corners=False)
        # concat
        out_hr = torch.cat((in_hr, out_lr_sr), dim=1)
        # high resolution
        out_hr = self.m_hr(out_hr).sigmoid_()
        
        return out_lr, out_hr


class Model(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', ch=3):  # model, input channels
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
            fuse_mode = cfg['backbone'][0][2] == 'Focus2'
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict
            fuse_mode = 'fuse' in cfg

        # Define model
        self.ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        ch = 6 if fuse_mode else 3
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Reconstructor) or isinstance(m, HEstimator):
            self.stride = m.stride
        else:
            self.stride = torch.tensor([4, 32])  # fake

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, profile=False, mode_align=False):
        # x1/m1: right image/mask, x2/m2: left image/mask. warp x2(left image) to x1(right image)
        x1, m1, x2, m2 = torch.split(x, [3, 1, 3, 1], dim=1)  # channel dimension
        
        if mode_align:
            module_range = (0, -1)
            feature1 = self.forward_once(x1, profile, module_range=module_range)
            feature2 = self.forward_once(x2, profile, module_range=module_range)
            return self.model[-1](feature1, feature2, x2, m2)
        else:
            x = torch.cat((x1, x2), dim=1)
            out = self.forward_once(x, profile)  # single-scale inference, train
            if not self.training:
                mask = ((m1 + m2) > 0).type_as(x)  # logical_or
                out = (out[0], out[1] * mask)  # higher resolution
            return out

    def forward_once(self, x, profile=False, module_range=None):
        y, dt = [], []  # outputs
        inputs = x
        modules = self.model if module_range is None else self.model[module_range[0]:module_range[1]]
        for m in modules:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

            if str(m.type) in 'models.yolo.Reconstructor':
                x = (*x, inputs)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            logger.info('%.1fms total' % sum(dt))
        return x

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    gd, gw = d['depth_multiple'], d['width_multiple']
    no = 3

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, Blur, CrossConv, BottleneckCSP,
                 C3, C3TR, Focus2, ConvWoBN, C3WoBN, FocusWoBN, ResBlock]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR, C3WoBN]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Add:
            c2 = ch[f[0]]
        elif m is Resizer:
            c2 = ch[f[0]] if isinstance(f, Iterable) else ch[f]
        elif m is Reconstructor:
            args.append([ch[x] for x in f])
        elif m in [HEstimator, HEstimatorOrigin, HEstimatorVariant]:
            args.append(ch[f])
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        elif m is nn.Identity:
            c2 = [ch[x] for x in f] if isinstance(f, Iterable) else ch[f]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 320, 320).to(device)
    # y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # logger.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
