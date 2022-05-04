import os
import cv2
import numpy as np
import random
import math
from time import time

import torch
from torch.utils.data import Dataset

from utils.general import _readline, make_divisible
from utils.torch_utils import torch_distributed_zero_first


def create_dataloader(path, imgsz, batch_size, mode='align', reg_mode='resize',
                      hyp=None, augment=False, rank=-1, world_size=1, workers=8):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagePairs(path, imgsz, augment=augment, hyp=hyp, mode=mode, reg_mode=reg_mode)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    # loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    loader = InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset, batch_size=batch_size, shuffle=augment,
                        num_workers=nw, sampler=sampler, pin_memory=True,
                        collate_fn=CollateFn(mode_align=mode == 'align')
                        )
    
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class CollateFn(object):
    def __init__(self, mode_align=True):
        self.mode_align = mode_align
    
    def __call__(self, batch):
        shapes = np.array([img.shape[:2] for img in batch])
        hm, wm = [make_divisible(int(shapes[:, i].max()), 32) for i in range(2)]
        if self.mode_align:
            hm, wm = max(hm, wm), max(hm, wm)  # square
    
        outs = []
        for i, img in enumerate(batch):
            h, w = shapes[i]
            dx, dy = (wm - w) // 2, (hm - h) // 2
            outs.append(np.pad(img, ((dy, hm - h - dy), (dx, wm - w - dx), (0, 0))))  # h,w,c
            
        imgs = np.stack(outs)
        imgs = imgs[..., ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, bhwc to bchw
        imgs = np.ascontiguousarray(imgs)
        
        return torch.from_numpy(imgs)


class LoadImagePairs(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, augment=False, hyp=None, mode='align', reg_mode='resize'):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.mode = mode
        self.path = path
        self.reg_mode = reg_mode  # `resize` or `crop`. Regularize the input images.
        
        with open(path, 'r') as f:
            self.img_files = f.read().splitlines()
        
        self.indices = range(len(self.img_files))  # number of images
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights
        
        # Load image
        # t0 = time()
        img1, img2, msk1, msk2 = self.load_data_(self.img_files[index])
        # t1 = time()
        
        hyp = self.hyp
        if self.augment:
            # Augment imagespace
            kwargs = {k: hyp[k] for k in ['degrees', 'translate', 'scale', 'shear', 'perspective']}
            # img1, img2 = random_perspective(img1, img2, **kwargs)
            if sum(kwargs.values()) > 0:
                if self.mode == 'align':
                    # disjoint
                    img1, msk1 = random_perspective(img1, msk1, **kwargs)
                    img2, msk2 = random_perspective(img2, msk2, **kwargs)
                    msk1, msk2 = msk1[..., np.newaxis], msk2[..., np.newaxis]
                else:
                    # joint
                    img1, img2 = np.concatenate((img1, msk1), axis=-1), np.concatenate((img2, msk2), axis=-1)
                    img2, img2 = random_perspective(img2, img2, **kwargs)
                    img1, msk1, img2, msk2 = img1[..., :3], img1[..., 3:], img2[..., :3], img2[..., 3:]
                    img1, msk1, img2, msk2 = [np.ascontiguousarray(_img) for _img in [img1, msk1, img2, msk2]]
            
            # Augment colorspace
            kwargs = {(k[-1] + 'gain'): hyp[k] for k in ['hsv_h', 'hsv_s', 'hsv_v']}
            if sum(kwargs.values()) > 0:
                if self.mode == 'align':
                    # disjoint
                    augment_hsv(img1, **kwargs)
                    augment_hsv(img2, **kwargs)
                else:
                    # joint
                    augment_hsv(img1, img2, **kwargs)
                    
            image = np.concatenate((msk1, img1, msk2, img2), axis=-1)
            
            # flip up-down
            if random.random() < hyp['flipud']:
                image = np.flipud(image)
            
            # flip left-right
            if random.random() < hyp['fliplr']:
                image = np.fliplr(image)
        else:
            image = np.concatenate((msk1, img1, msk2, img2), axis=-1)

        # t2 = time()
        # print('load: %.3fms, augment: %.3fms, total: %.3fms' % ((t1 - t0) * 1000, (t2 - t1) * 1000, (t2 - t0) * 1000))
        
        return image  # ch=1+3+1+3=8, hwc

    def load_data_(self, path):
        if self.mode == 'align':
            img1 = cv2.imread(path)
            img2 = cv2.imread(path.replace('input1', 'input2'))
            new_size = (self.img_size, self.img_size)
            if tuple(img1.shape[:2]) != new_size or tuple(img2.shape[:2]) != new_size:
                interpolation = cv2.INTER_AREA  # cv2.INTER_LINEAR cv2.INTER_AREA
                img1 = cv2.resize(img1, new_size, interpolation=interpolation)
                img2 = cv2.resize(img2, new_size, interpolation=interpolation)
            msk1 = np.zeros((*new_size, 1), dtype=np.uint8) + 255
            msk2 = np.zeros((*new_size, 1), dtype=np.uint8) + 255
        
            # if self.augment and random.random() < 0.5:
            #     # change the warping direction: img2 to img1
            #     img1, img2, msk1, msk2 = img2, img1, msk2, msk1
        else:
            path = path.replace('UDIS-D', 'UDIS-D/warp').replace('input1/', '')
            base_name = os.path.basename(path)
            img_name = base_name[:-4]
            img1 = cv2.imread(path.replace(base_name, f'{img_name}_warp1.jpg'))
            img2 = cv2.imread(path.replace(base_name, f'{img_name}_warp2.jpg'))
            msk1 = cv2.imread(path.replace(base_name, f'{img_name}_mask1.png'))
            msk2 = cv2.imread(path.replace(base_name, f'{img_name}_mask2.png'))
            if self.img_size > 0:
                if self.reg_mode == 'resize':
                    new_size = (self.img_size, self.img_size)
                    interpolation = cv2.INTER_LINEAR
                    img1, img2, msk1, msk2 = [cv2.resize(_img, new_size, interpolation=interpolation)
                                              for _img in [img1, img2, msk1, msk2]]
                elif self.reg_mode == 'crop':
                    height, width, _ = img1.shape
                    msk1, msk2 = msk1[..., :1], msk2[..., :1]
                    overlap = cv2.bitwise_and(msk1[..., 0], msk2[..., 0])
                    y, x = np.nonzero(overlap)
                    xc, yc = np.median(x).astype(np.int32), np.median(y).astype(np.int32)
                    x10, y10, x20, y20 = xc - self.img_size // 2, yc - self.img_size // 2, \
                                         xc + self.img_size // 2, yc + self.img_size // 2
                    x1, y1, x2, y2 = x10.clip(0), y10.clip(0), x20.clip(0, width), y20.clip(0, height)
                    pad_x1, pad_y1, pad_x2, pad_y2 = x1 - x10, y1 - y10, x20 - x2, y20 - y2
                    imgs = np.concatenate((img1, msk1, img2, msk2), axis=-1)
                    imgs = imgs[y1:y2, x1:x2]
                    imgs = np.pad(imgs, ((pad_y1, pad_y2), (pad_x1, pad_x2), (0, 0)))
                    img1, msk1, img2, msk2 = [np.ascontiguousarray(img_) for img_ in np.split(imgs, [3, 4, 7], axis=-1)]
                else:
                    raise ValueError('Invalid regularization mode: ', self.reg_mode)
            msk1, msk2 = msk1[..., :1], msk2[..., :1]
            
        return img1, img2, msk1, msk2
        # return img2, img1, msk2, msk1


def random_perspective(image, mask=None, degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = image.shape[0] + border[0] * 2  # shape(h,w,c)
    width = image.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -image.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -image.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            image = cv2.warpPerspective(image, M, dsize=(width, height), borderValue=(0, 0, 0))
            mask = cv2.warpPerspective(mask, M, dsize=(width, height), borderValue=(0, 0, 0)) \
                if mask is not None else None
        else:  # affine
            image = cv2.warpAffine(image, M[:2], dsize=(width, height), borderValue=(0, 0, 0))
            mask = cv2.warpAffine(mask, M[:2], dsize=(width, height), borderValue=(0, 0, 0)) \
                if mask is not None else None

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    return (image, mask) if mask is not None else image


def augment_hsv(img, img2=None, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
    
    if img2 is not None:
        hue, sat, val = cv2.split(cv2.cvtColor(img2, cv2.COLOR_BGR2HSV))
        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img2)  # no return needed


def color_balance(img1, img2, msk1=None, msk2=None, normalized=False):
    if msk1 is None or msk2 is None:
        msk1 = np.ones_like(img1[..., :1])
        msk2 = np.ones_like(img2[..., :1])
    if normalized:
        img1, img2 = img1 * 255., img2 * 255.

    idx1, idx2 = msk1[..., 0] > 0, msk2[..., 0] > 0
    r = ((img2[idx2] / 255.).mean(axis=0) / (img1[idx1] / 255.).mean(axis=0)).reshape(1, 1, 3)
    img1 = (((img1 / 255.) * r).clip(0, 1) * 255.).astype(np.uint8)
    return img1
