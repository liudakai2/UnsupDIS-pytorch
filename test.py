import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import torch

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import check_dataset, check_file, check_img_size, set_logging, increment_path, img_torch2numpy, \
    check_align_input, check_align_output
from utils.torch_utils import select_device, time_synchronized
from utils.loss import ComputeAlignLoss, ComputeFuseLoss


@torch.no_grad()
def test(data, weights=None, batch_size=32, imgsz=640, model=None, dataloader=None, half_precision=True,
         compute_loss=None, save_dir=Path(''), opt=None, mode_align=False):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
    
    else:  # called directly
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)
        
        # Directories
        save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir
        
        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size
        compute_loss = ComputeAlignLoss(model) if opt.mode == 'align' else ComputeFuseLoss(model)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()
    else:
        model.float()

    # Configure
    model.eval()
    if isinstance(data, str):
        with open(data) as f:
            data = yaml.safe_load(f)
    check_dataset(data)  # check
    mode_align = mode_align or (opt and opt.mode == 'align')
    
    # Dataloader
    if not training:
        if device.type != 'cpu':
            dummy = torch.zeros(1, 8 if mode_align else 6, imgsz, imgsz).to(device).type_as(next(model.parameters()))
            model(dummy, mode_align=mode_align)  # run once
        task = opt.task if opt.task in ('train', 'val', 'test') else 'test'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, mode=opt.mode, reg_mode=opt.reg_mode, augment=False)[0]

    seen, t0, crashed = 0, 0, 0
    if mode_align:
        s = ('%20s' + '%10s' * 6) % ('Images', 'loss1', 'loss2', 'loss3', 'total', 'PSNR', 'SSIM')
        psnr_list, ssim_list = [], []
    else:
        s = ('%20s' + '%10s' * 6) % ('Images', 'cont_lr', 'seam_lr', 'cont_hr', 'seam_hr', 'consist', 'total')
    losses = torch.zeros(4 if mode_align else 6, device=device)
    eps = 0.01
    for batch_i, imgs in enumerate(tqdm(dataloader, desc=s)):
        imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
        if half:
            imgs = imgs.half()
            
        seen += imgs.shape[0]

        # Run model
        t = time_synchronized()
        pred = model(imgs, mode_align=mode_align)  # training outputs
        t0 += time_synchronized() - t
        # check_align_input(imgs, _exit=False, normalized=True)
        # check_align_output(*pred[1:], _exit=True)
        if compute_loss:
            losses += compute_loss(pred, imgs)[1]
        
        if mode_align:
            target_imgs = imgs[:, :3, ...]
            warped_imgs, warped_ones = pred[1][-1], pred[2][-1]
            for i, warped_mask in enumerate(warped_ones):
                warped_mask = (warped_mask > eps).all(axis=0, keepdims=True).float()
                if warped_mask.sum() == 0:
                    crashed += 1
                    psnr_list.append(0)
                    ssim_list.append(0)
                    check_align_input(imgs, _exit=False, normalized=True)
                    check_align_output(*pred[1:], _exit=True)
                    continue
                target_img = img_torch2numpy(target_imgs[i] * warped_mask)
                warped_img = img_torch2numpy(warped_imgs[i] * warped_mask)
                psnr_list.append(peak_signal_noise_ratio(target_img, warped_img, data_range=255))
                ssim_list.append(structural_similarity(target_img, warped_img, data_range=255, multichannel=True))
            # if batch_i == 0:
            #     check_align_input(imgs, _exit=False, normalized=True)
            #     check_align_output(*pred[1:], _exit=False)

    losses /= (batch_i + 1)
    measures = [np.mean(psnr_list), np.mean(ssim_list)] if mode_align else []
    pf = '%20i' + '%10.3g' * 6  # print format
    print(pf % (seen, *losses, *measures))
    t = t0 / seen * 1E3
    if not training:
        print('Speed: %.1f ms inference per %gx%g image at batch-size %g' % (t, imgsz, imgsz, batch_size))
        print(f'total {seen}, crashed {crashed}')
    
    return losses.cpu().tolist() + measures


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--reg-mode', default='resize', choices=['resize', 'crop'], help='image regularization')
    parser.add_argument('--task', default='val', help='train, val, test')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--mode', default='align', choices=['align', 'fuse'], help='task mode, align or fuse')
    opt = parser.parse_args()
    opt.data = check_file(opt.data)  # check file
    print(opt)

    test(opt.data, opt.weights, opt.batch_size, opt.img_size, half_precision=opt.half, opt=opt)
