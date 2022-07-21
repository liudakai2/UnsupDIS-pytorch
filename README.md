## A pytorch-reimplementation for [Unsupervised Deep Image Stitching: Reconstructing Stitched Features to Images](https://arxiv.org/pdf/2106.12859.pdf).

The official implementation is [here](https://github.com/nie-lang/UnsupervisedDeepImageStitching) with tensorflow 1.x. The stitching pipeline referred to [UnsupDIS](https://github.com/nie-lang/UnsupervisedDeepImageStitching) and the networks and code organizations utilized [YOLOv5](https://github.com/ultralytics/yolov5). Both of them are excellent works.

This repo allows you to finish the whole training process (including alignment and reconstruction) within 1 day. This repo makes it possible to be a real-time application during inference.

## Results
![image](https://github.com/liudakai2/UnsupDIS-pytorch/blob/main/assets/sample.jpg)


## Pretrained Checkpoints

<!-- [assets]: https://github.com/liudakai2/UnsupDIS-pytorch/releases -->

<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
.tg .tg-nrix{text-align:center;vertical-align:middle}
</style> -->
<table class="tg" style='text-align:center;vertical-align:middle'>
<thead>
  <tr>
    <th class="tg-9wq8" colspan="2" rowspan="2">Model</th>
    <th class="tg-9wq8" colspan="3">Warped MS COCO</th>
    <th class="tg-nrix" colspan="2">UDIS-D</th>
    <th class="tg-nrix" rowspan="2">Param(M)</th>
    <th class="tg-nrix" rowspan="2">GFLOPs</th>
  </tr>
  <tr>
    <th class="tg-9wq8">PSNR</th>
    <th class="tg-9wq8">SSIM</th>
    <th class="tg-9wq8">RMSE</th>
    <th class="tg-nrix">PSNR</th>
    <th class="tg-nrix">SSIM</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="4">Align</td>
    <td class="tg-9wq8"><a href="https://github.com/liudakai2/UnsupDIS-pytorch/releases">origin.tf</a></td>
    <td class="tg-9wq8">-</td>
    <td class="tg-9wq8">-</td>
    <td class="tg-9wq8">2.0239</td>
    <td class="tg-nrix">23.80</td>
    <td class="tg-nrix">0.7929</td>
    <td class="tg-nrix" rowspan="2">180</td>
    <td class="tg-nrix" rowspan="2">14.3</td>
  </tr>
  <tr>
    <td class="tg-9wq8"><a href="https://github.com/liudakai2/UnsupDIS-pytorch/releases">origin</a></td>
    <td class="tg-9wq8">33.95</td>
    <td class="tg-9wq8">0.9481</td>
    <td class="tg-9wq8">2.0695</td>
    <td class="tg-nrix">26.34</td>
    <td class="tg-nrix">0.8589</td>
  </tr>
  <tr>
    <td class="tg-nrix"><a href="https://github.com/liudakai2/UnsupDIS-pytorch/releases">yolo</a></td>
    <td class="tg-nrix">36.64</td>
    <td class="tg-nrix">0.9657</td>
    <td class="tg-nrix" style='font-weight:bold'>1.7241</td>
    <td class="tg-nrix" style='font-weight:bold'>26.53</td>
    <td class="tg-nrix" style='font-weight:bold'>0.8641</td>
    <td class="tg-nrix">15</td>
    <td class="tg-nrix">14.5</td>
  </tr>
  <tr>
    <td class="tg-nrix"><a href="https://github.com/liudakai2/UnsupDIS-pytorch/releases">variant</a></td>
    <td class="tg-nrix" style='font-weight:bold'>37.33</td>
    <td class="tg-nrix" style='font-weight:bold'>0.9704</td>
    <td class="tg-nrix">1.7614</td>
    <td class="tg-nrix" style='font-weight:bold'>26.53</td>
    <td class="tg-nrix" style='font-weight:bold'>0.8622</td>
    <td class="tg-nrix" style='font-weight:bold'>9.7</td>
    <td class="tg-nrix" style='font-weight:bold'>12.3</td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="2">Fuse</td>
    <td class="tg-nrix">origin</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">8</td>
    <td class="tg-nrix">605.3</td>
  </tr>
  <tr>
    <td class="tg-nrix"><a href="https://github.com/liudakai2/UnsupDIS-pytorch/releases">yolo</a></td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">-</td>
    <td class="tg-nrix">4.4</td>
    <td class="tg-nrix" style='font-weight:bold'>74.8</td>
  </tr>
</tbody>
</table>


## Installation
[**Python>=3.6**](https://www.python.org/) is required with all
[requirements.txt](requirements.txt) installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):

```bash
python3 -m pip install -r requirements.txt
```


## Data Preparation
Download the [UDIS-D](https://drive.google.com/drive/folders/1kC7KAULd5mZsqaWnY3-rSbQLaZ7LujTY?usp=sharing) and [WarpedCOCO](https://pan.baidu.com/s/1MVn1VFs_6-9dNRVnG684og) (code: 1234), and
make soft-links to the data directories:

```bash
ln -sf /path/to/UDIS-D UDIS-D
ln -sf /path/to/WarpedCOCO WarpedCOCO
```

Make sure the images are organized as follows:

```bash
UDIS-D/train/input1/000001.jpg  UDIS-D/train/input2/000001.jpg  UDIS-D/test/input1/000001.jpg  UDIS-D/test/input2/000001.jpg
WarpedCOCO/training/input1/000001.jpg  WarpedCOCO/training/input2/000001.jpg  WarpedCOCO/testing/input1/000001.jpg  WarpedCOCO/testing/input2/000001.jpg
```


## Training, Testing, and Inference
Run the commands below to go through the whole process of unsupervised deep image stitching. Some alternative commands are displayed in [main.sh](main.sh).

Download the pretrained backbones ([YOLOv5s](https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt), [YOLOv5m](https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5m.pt), [YOLOv5l](https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5l.pt), [YOLOv5x](https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5x.pt)) and put them to the `weights/` directory first. You can modify the `depth_multiple` and `width_multiple` in `models/*.yaml` to choose which backbone to use. 

#### Step 1 (Alignment): Unsupervised pre-training on Stitched MS-COCO

```bash
python3 train.py --data data/warpedcoco.yaml --hyp data/hyp.align.scratch.yaml --cfg models/align.yaml --weights weights/yolov5l.pt --batch-size 64 --img-size 128 --epochs 50 --adam --device 0 --mode align
mv runs/train/exp weights/align/warpedcoco
```

#### Step 2 (Alignment): Unsupervised finetuning on UDIS-D

```bash
python3 train.py --data data/udis.yaml --hyp data/hyp.align.finetune.yaml --cfg models/align.yaml --weights weights/align/warpedcoco/weights/best.pt --batch-size 64 --img-size 128 --epochs 30 --adam --device 0 --mode align
mv runs/train/exp weights/align/udis
```

#### Step 3 (Alignment): Evaluating and visualizing the alignment results

```bash
(RMSE) python3 inference_align.py --source data/warpedcoco.yaml --weights weights/align/warpedcoco/weights/best.pt --task val --rmse
(PSNR) python3 test.py --data data/warpedcoco.yaml --weights weights/align/warpedcoco/weights/best.pt --batch-size 64 --img-size 128 --task val --device 0 --mode align
(PSNR) python3 test.py --data data/udis.yaml --weights weights/align/udis/weights/best.pt --batch-size 64 --img-size 128 --task val --device 0 --mode align
(PLOT) python3 inference_align.py --source data/udis.yaml --weights weights/align/udis/weights/best.pt --task val --visualize
rm -r runs/infer/ runs/test/
```

#### Step 4 (Alignment): Generating the coarsely aligned image pairs

```bash
python3 inference_align.py --source data/udis.yaml --weights weights/align/udis/weights/best.pt --task train
python3 inference_align.py --source data/udis.yaml --weights weights/align/udis/weights/best.pt --task test
mkdir UDIS-D/warp
mv runs/infer/exp UDIS-D/warp/train
mv runs/infer/exp2 UDIS-D/warp/test
```

#### Step 5 (Reconstruction): Training the reconstrction model on UDIS-D

```bash
python3 train.py --data data/udis.yaml --hyp data/hyp.fuse.scratch.yaml --cfg models/fuse.yaml --weights weights/yolov5m.pt --batch-size 12 --img-size 640 --epochs 30 --adam --device 0 --mode fuse --reg-mode crop
mv runs/train/exp weights/fuse/udis
```

#### Step 6 (Reconstruction): Generating the finally stitched results

```bash
python3 inference_fuse.py --weights weights/fuse/udis/weights/best.pt --source data/udis.yaml --task test --half --img-size 640 --reg-mode crop
```

## TODO
- [ ] **Fuse Optimization**: the hyp-params and  loss functions are not perfect right now.

## Comparison and Discussion

In order to improve the flexibility and speed up the training process, we made this reimplementation in pytorch. We also adjusted the networks, loss functions, data augmentation, and a considerable part of hyper-parameters. Taking the network as an example, in the alignment phase we replace the `CostVolume` module with a simple `Concat` module, of which the former is either time-consuming or memory-consuming, and even leads to divergence. And the alignment training process may be broken for some unknown reasons. This repository is far
away from perfect, and I hope you can assist me to complete this project. Contact me if you have any problems or suggestions -- kail@zju.edu.cn.