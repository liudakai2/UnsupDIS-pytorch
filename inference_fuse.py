import argparse
import time
import yaml
from pathlib import Path
from os.path import basename
from tqdm import tqdm

import cv2
import numpy as np
import torch

from models.experimental import attempt_load
from utils.general import increment_path, check_img_size, set_logging, check_dataset, img_numpy2torch, img_torch2numpy
from utils.torch_utils import select_device, time_synchronized


def parse_source(source, task, imgsz, reg_mode):
    if source.endswith('.yaml'):
        with open(source, 'r') as f:
            data = yaml.safe_load(f)
        check_dataset(data)
        with open(data[task], 'r') as f:
            source = f.read().splitlines()
    else:
        raise ValueError(f"invalid source: {source}")
    
    for path in source:
        path = path.replace('UDIS-D', 'UDIS-D/warp').replace('input1/', '')
        base_name = basename(path)
        img_name = base_name[:-4]
        img1 = cv2.imread(path.replace(base_name, f'{img_name}_warp1.png'))
        img2 = cv2.imread(path.replace(base_name, f'{img_name}_warp2.png'))
        msk1 = cv2.imread(path.replace(base_name, f'{img_name}_mask1.png'))
        msk2 = cv2.imread(path.replace(base_name, f'{img_name}_mask2.png'))
        image_raw = np.concatenate((img1, msk1[..., :1], img2, msk2[..., :1]), axis=-1)
        height, width, _ = img1.shape
        xyxy_raw = [0, 0, width, height]
        xyxy_inp = [0, 0, width, height]
        if imgsz > 0:
            if reg_mode == 'resize':
                img1, img2, msk1, msk2 = [cv2.resize(_img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
                                          for _img in [img1, img2, msk1, msk2]]
            elif reg_mode == 'crop':
                msk1, msk2 = msk1[..., :1], msk2[..., :1]
                overlap = cv2.bitwise_and(msk1[..., 0], msk2[..., 0])
                y, x = np.nonzero(overlap)
                xc, yc = np.median(x).astype(np.int32), np.median(y).astype(np.int32)
                x10, y10, x20, y20 = xc - imgsz // 2, yc - imgsz // 2, xc + imgsz // 2, yc + imgsz // 2
                x1, y1, x2, y2 = x10.clip(0), y10.clip(0), x20.clip(0, width), y20.clip(0, height)
                pad_x1, pad_y1, pad_x2, pad_y2 = x1 - x10, y1 - y10, x20 - x2, y20 - y2
                imgs = np.concatenate((img1, msk1, img2, msk2), axis=-1)
                imgs = imgs[y1:y2, x1:x2]
                imgs = np.pad(imgs, ((pad_y1, pad_y2), (pad_x1, pad_x2), (0, 0)))
                img1, msk1, img2, msk2 = [np.ascontiguousarray(img_) for img_ in np.split(imgs, [3, 4, 7], axis=-1)]
                xyxy_raw = [x1, y1, x2, y2]
                xyxy_inp = [pad_x1, pad_y1, pad_x1 + x2 - x1, pad_y1 + y2 - y1]
            else:
                raise NotImplemented(f'Image regularization mode {reg_mode} is not supported.')
        msk1, msk2 = msk1[..., :1], msk2[..., :1]
        image = np.concatenate((msk1, img1, msk2, img2), axis=-1)
        
        yield path, image_raw, img_numpy2torch(image), (xyxy_raw, xyxy_inp)
        
        
def post_process(pred, img_raw, xyxy_inp, xyxy_raw, reg_mode):
    if reg_mode == 'resize':
        if pred.shape[:2] != img_raw.shape[:2]:
            return cv2.resize(pred, img_raw.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
    elif reg_mode == 'crop':
        xi1, yi1, xi2, yi2 = xyxy_inp
        xr1, yr1, xr2, yr2 = xyxy_raw
        delta = 5
        xi1, yi1, xi2, yi2 = xi1 + delta, yi1 + delta, xi2 - delta, yi2 - delta
        xr1, yr1, xr2, yr2 = xr1 + delta, yr1 + delta, xr2 - delta, yr2 - delta
    
        out = img_raw.copy()
        out[yr1:yr2, xr1:xr2] = pred[yi1:yi2, xi1:xi2]
        return out
    else:
        raise NotImplementedError(reg_mode)


def img_overlay(imgs):
    img1, msk1, img2, msk2 = [np.ascontiguousarray(_img) for _img in np.split(imgs, (3, 4, 7), axis=-1)]
    overlap = ((msk1 / 255.) * (msk2 / 255.) > 0).astype(np.uint8)
    image = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
    # brightness compensation
    image = cv2.addWeighted(image * overlap, 1.0, image * (1 - overlap), 2.0, 0)
    
    return image


@torch.no_grad()
def detect():
    source, weights, imgsz, task, reg_mode = opt.source, opt.weights, opt.img_size, opt.task, opt.reg_mode

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = opt.half and device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(int(imgsz), s=stride) if imgsz > 1 else imgsz  # check img_size
    if half:
        model.half()  # to FP16

    # Run inference
    dataloader = parse_source(source, task, imgsz, reg_mode)
    
    count = 0
    for path, imgs_raw, imgs, (xyxy_raw, xyxy_inp) in dataloader:
        count += 1
        imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
        if half:
            imgs = imgs.half()
        imgs = imgs.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        preds_lr, preds = model(imgs, mode_align=False)  # inference and training outputs
        t2 = time_synchronized()

        # post_process(preds, imgs, seam_extractor)
        pred = img_torch2numpy(preds[0])
        overlapped = img_overlay(imgs_raw)
        pred = post_process(pred, overlapped, xyxy_inp, xyxy_raw, reg_mode)
        t3 = time_synchronized()

        img_name = basename(path)[:-4]
        cv2.imwrite(str(save_dir / (img_name + '.png')), pred)
        # (optional)
        # cv2.imwrite(str(save_dir / (img_name + '_raw.jpg')), overlapped)
        # if count >= 20:
        #     break
        
        print(f'{str(save_dir)}/{count:06d} Done. ({t3 - t1:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--task', default='val', help='train, val, test')
    parser.add_argument('--img-size', type=float, default=-1, help='inference size (pixels) or scale ratio')
    parser.add_argument('--reg-mode', default='resize', choices=['resize', 'crop'], help='image regularization')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/infer', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    print(opt)

    detect()
