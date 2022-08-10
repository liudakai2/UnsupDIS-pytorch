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
from utils.general import increment_path, check_dataset, check_img_size, set_logging, img_numpy2torch, \
    check_align_input, check_align_output, img_torch2numpy
from utils.torch_utils import select_device, time_synchronized
from utils.stitching import Stitching_Domain_STN


def parse_source(source, imgsz):
    is_coco = 'coco' in source
    if source.endswith('.yaml'):
        with open(source, 'r') as f:
            data = yaml.safe_load(f)
        check_dataset(data)
        with open(data[opt.task], 'r') as f:
            source = f.read().splitlines()
    else:
        raise ValueError(f"invalid source: {source}")
    
    for path in source:
        img_left = cv2.imread(path)
        img_right = cv2.imread(path.replace('input1', 'input2'))

        # TODO: swap img1 and img2 (default or optional)
        if is_coco and opt.rmse:
            img_left, img_right = img_right, img_left
        
        height, width = img_right.shape[:2]
        size_tensor = torch.tensor([width, height])
        if img_left.shape != img_right.shape:
            raise NotImplementedError
            # img_left = cv2.resize(img_left, (width, height))
    
        if (width, height) != (imgsz, imgsz):
            img1 = cv2.resize(img_left, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img_right, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
        else:
            img1, img2 = img_left, img_right
        msk1 = np.ones_like(img1[..., :1])
        msk2 = np.ones_like(img2[..., :1])
        
        image = np.concatenate((msk1, img1, msk2, img2), axis=-1)
        imgs_raw = np.concatenate((img_left, img_right), axis=-1)
        
        yield path, img_numpy2torch(imgs_raw), img_numpy2torch(image), size_tensor


@torch.no_grad()
def infer():
    source, weights, imgsz = opt.source, opt.weights, opt.img_size

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)
    # warp1_dir, mask1_dir, warp2_dir, mask2_dir = \
    #     save_dir / 'warp1', save_dir / 'mask1', save_dir / 'warp2', save_dir / 'mask2'
    # if not opt.visualize:  # output mode
    #     for d in [warp1_dir, mask1_dir, warp2_dir, mask2_dir]:
    #         d.mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = opt.half and device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    model.mode_align = True
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(int(imgsz), s=stride) if imgsz > 1 else imgsz  # check img_size
    if half:
        model.half()  # to FP16

    # Run inference
    dataloader = parse_source(source, imgsz)
    
    count, crashed, rmse = 0, 0, []
    for path, imgs_raw, imgs, size_tensor in tqdm(dataloader):
        count += 1
        imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
        imgs_raw = imgs_raw.to(device, non_blocking=True).float() / 255.0
        size_tensor = size_tensor.to(device)
        if half:
            imgs = imgs.half()
        imgs = imgs.unsqueeze(0)
        imgs_raw = imgs_raw.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        vertices_offsets, warped_imgs, warped_ones = model(imgs)
        t2 = time_synchronized()
        # check_align_input(imgs, _exit=False, normalized=True)
        # check_align_output(warped_imgs, warped_ones, _exit=True)

        resized_shift = vertices_offsets * size_tensor.repeat(4).reshape(1, 8, 1) / imgsz
        output = Stitching_Domain_STN(imgs_raw, size_tensor, resized_shift)
        mask_left, warped_left, mask_right, warped_right = \
            np.split(img_torch2numpy(output[0]), (1, 4, 5), axis=-1)

        t3 = time_synchronized()
        
        # print(str(save_dir), end='/')
        # print(f'{count:06d} Done. '
        #       f'(Inference: {t2 - t1:.3f}s. Postprocessing: {t3 - t2:.3f}s. Total: {t3 - t1:.3f}s.)')

        if opt.visualize:
            # for visualization
            cv2.imwrite(str(save_dir / ('%06d_warped_left.jpg' % count)), warped_left)
            cv2.imwrite(str(save_dir / ('%06d_warped_right.jpg' % count)), warped_right)
            cv2.imwrite(str(save_dir / ('%06d_warped_merge.jpg' % count)),
                        cv2.addWeighted(warped_right, 0.5, warped_left, 0.5, 0))
            # if count >= 100:
            #     break
        if opt.rmse:
            assert 'coco' in path.lower()
            # this shift transform img2 to img1! fxxk.
            pred_shift = resized_shift[0].cpu().numpy().reshape(4, 2)
            gt_shift = np.load(path.replace('input1', 'shift').replace('.jpg', '.npy')).reshape(4, 2)
            
            # org = np.array([0, 0, imgsz, 0, 0, imgsz, imgsz, imgsz]).reshape(4, 2).astype(np.float32)
            # dst = org + gt_shift
            # perspective_matrix_inv = cv2.getPerspectiveTransform(dst, org)
            # org = np.concatenate((org, np.ones((4, 1), dtype=np.float32)), axis=1)
            # dst = np.concatenate([perspective_matrix_inv @ org[i, :] for i in range(4)]).reshape(4, 3)
            # dst /= dst[:, -1:]
            # gt_shift = (dst - org)[:, :2].reshape(8, 1)
            
            rmse.append(np.sqrt(np.power(pred_shift - gt_shift, 2.).mean()))
            # print(rmse[-1])
            if rmse[-1] > 100:
                print(rmse[-1])
                print(pred_shift.reshape(-1))
                print(gt_shift.reshape(-1))
                break
        if not opt.visualize and not opt.rmse:
            # for image fusion
            img_name = basename(path)[:-4]
            # cv2.imwrite(str(warp1_dir / (img_name + '.jpg')), warped_left)
            # cv2.imwrite(str(mask1_dir / (img_name + '.png')), mask_left)
            # cv2.imwrite(str(warp2_dir / (img_name + '.jpg')), warped_right)
            # cv2.imwrite(str(mask2_dir / (img_name + '.png')), mask_right)
            cv2.imwrite(str(save_dir / (img_name + '_warp1.jpg')), warped_left)
            cv2.imwrite(str(save_dir / (img_name + '_mask1.png')), mask_left)
            cv2.imwrite(str(save_dir / (img_name + '_warp2.jpg')), warped_right)
            cv2.imwrite(str(save_dir / (img_name + '_mask2.png')), mask_right)
                
    print("RMSE: %.4f" % (np.mean(rmse))) if len(rmse) > 0 else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--task', default='val', help='train, val, test')
    parser.add_argument('--img-size', type=int, default=128, help='inference size (pixels) or scale ratio')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default='runs/infer', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--visualize', action='store_true', help='organize outputs for visualization')
    parser.add_argument('--rmse', action='store_true', help='calculate the 4-pt RMSE')
    opt = parser.parse_args()
    print(opt)

    infer()
