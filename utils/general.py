import os
import numpy as np
import cv2
import random
import math
import logging
import glob
import re
import urllib
from pathlib import Path

import torch

from utils.torch_utils import init_torch_seeds

# Settings
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8))  # NumExpr max threads


def _readline(path, _split=False, sep=None, map_func=None):
    with open(path, 'r') as f:
        line = f.readline()
    if _split:
        line = line.split(sep)
        if map_func is not None:
            line = list(map(map_func, line))
    
    return line


def img_torch2numpy(img, reverse_channel=True, normalized=True):
    batch_flag = len(tuple(img.shape)) == 4
    imgs = [img] if not batch_flag else img
    imgs_np = []
    for img in imgs:
        img = img.detach().cpu().numpy()
        if reverse_channel:
            img = img[::-1, ...]
        img = img.transpose(1, 2, 0)
        img = np.ascontiguousarray(img)
        if normalized:
            img = (img.clip(0, 1) * 255.).astype(np.uint8)
        
        imgs_np.append(img)
    
    if batch_flag:
        return np.stack(imgs_np)
    else:
        return imgs_np[0]


def img_numpy2torch(img):
    batch_flag = len(tuple(img.shape)) == 4
    imgs = [img] if not batch_flag else img
    imgs_torch = []
    for img in imgs:
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        imgs_torch.append(torch.from_numpy(img))
    
    if batch_flag:
        return torch.stack(imgs_torch)
    else:
        return imgs_torch[0]


def check_align_input(imgs, _exit=True, normalized=False):
    if not os.path.exists('./tmp'):
        os.mkdir('./tmp')
    else:
        for path in glob.glob('tmp/*'):
            os.remove(path)
    for j, img in enumerate(imgs):
        img = img_torch2numpy(img, normalized=normalized)
        cv2.imwrite('tmp/%02d_left.jpg' % j, img[..., 1:4])
        cv2.imwrite('tmp/%02d_right.jpg' % j, img[..., 5:8])
    exit(123) if _exit else None


def check_align_output(warped_imgs, warped_ones, _exit=True):
    for i, imgs in enumerate(warped_imgs):
        for j, img in enumerate(imgs):
            img = img_torch2numpy(img, normalized=True)
            # cv2.imwrite('tmp/%02d_left_warped%d.jpg' % (j, i), img)
            # msk = img_torch2numpy(warped_ones[i][j], normalized=True)
            # cv2.imwrite('tmp/%02d_left_warped%d_mask.jpg' % (j, i), msk)
            if i == len(warped_imgs) - 1 and True:
                cv2.imwrite('tmp/%02d_left_warped.jpg' % j, img)
                img2 = cv2.imread('tmp/%02d_right.jpg' % j)
                img = cv2.addWeighted(img, 0.5, img2, 0.5, 0)
                cv2.imwrite('tmp/%02d_merge_warped.jpg' % j, img)
    exit(123) if _exit else None


def check_fuse_input(imgs):
    for j, img in enumerate(imgs):
        img = img.numpy()[::-1, ...].transpose(1, 2, 0)
        img = np.ascontiguousarray(img)
        cv2.imwrite('tmp/%02d_left.png' % j, img[..., :1])
        cv2.imwrite('tmp/%02d_left.jpg' % j, img[..., 1:4])
        cv2.imwrite('tmp/%02d_right.png' % j, img[..., 4:5])
        cv2.imwrite('tmp/%02d_right.jpg' % j, img[..., 5:])
    exit(123)


def set_logging(rank=-1, verbose=True):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if (verbose and rank in [-1, 0]) else logging.WARN)


def get_latest_run(search_dir='.'):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=''):
    from copy import copy
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rc('font', **{'size': 11})
    matplotlib.use('Agg')  # for writing to files only
    
    # Plot LR simulating training for full epochs
    optimizer, scheduler = copy(optimizer), copy(scheduler)  # do not modify originals
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.savefig(Path(save_dir) / 'LR.png', dpi=200)
    plt.close()


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def check_dataset(dict):
    # Download dataset if not found locally
    val, s = dict.get('val'), dict.get('download')
    if val and len(val):
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            print('\nWARNING: Dataset not found, nonexistent paths: %s' % [str(x) for x in val if not x.exists()])
            if s and len(s):  # download script
                if s.startswith('http') and s.endswith('.zip'):  # URL
                    f = Path(s).name  # filename
                    print(f'Downloading {s} ...')
                    torch.hub.download_url_to_file(s, f)
                    r = os.system(f'unzip -q {f} -d ../ && rm {f}')  # unzip
                elif s.startswith('bash '):  # bash script
                    print(f'Running {s} ...')
                    r = os.system(s)
                else:  # python script
                    r = exec(s)  # return None
                print('Dataset autodownload %s\n' % ('success' if r in (0, None) else 'failure'))  # print result
            else:
                raise Exception('Dataset not found.')


def check_img_size(img_size, s=32, warn=True):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size and warn:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


def check_file(file):
    # Search/download file (if necessary) and return path
    file = str(file)  # convert to str()
    if Path(file).is_file() or file == '':  # exists
        return file
    elif file.startswith(('http://', 'https://')):  # download
        url, file = file, Path(urllib.parse.unquote(str(file))).name  # url, file (decode '%2F' to '/' etc.)
        file = file.split('?')[0]  # parse authentication https://url.com/file.txt?auth...
        print(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, file)
        assert Path(file).exists() and Path(file).stat().st_size > 0, f'File download failed: {url}'  # check
        return file
    else:  # search
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), f'File not found: {file}'  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path
