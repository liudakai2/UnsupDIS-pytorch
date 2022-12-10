## A pytorch-reimplementation for [Unsupervised Deep Image Stitching: Reconstructing Stitched Features to Images](https://arxiv.org/pdf/2106.12859.pdf).

The official implementation is [here](https://github.com/nie-lang/UnsupervisedDeepImageStitching) with tensorflow 1.x. The stitching pipeline referred to [UnsupDIS](https://github.com/nie-lang/UnsupervisedDeepImageStitching) and the networks and code organizations utilized [YOLOv5](https://github.com/ultralytics/yolov5). Both of them are excellent works.

This repo allows you to finish the whole training process (including alignment and reconstruction) within 1 day. This repo makes it possible to be a real-time application during inference.

## Results
![image](https://github.com/liudakai2/UnsupDIS-pytorch/blob/main/assets/sample.jpg)


## Pretrained Checkpoints

[assets]: https://github.com/liudakai2/UnsupDIS-pytorch/releases

|Model |COCO<br>PSNR |COCO<br>SSIM |COCO<br>RMSE |UDIS<br>PSNR |UDIS<br>SSIM |Params(M) |GFLOPs
|---                       |:-:       |:-:        |:-:        |:-:       |:-:        |:-:     |:-:
|align-origin.tf           |-         |-          |2.0239     |23.80     |0.7929     |180.0   |14.3
|align-origin              |33.95     |0.9481     |2.0695     |26.34     |0.8589     |180.0   |14.3
|[align-yolo][assets]      |36.64     |0.9657     |**1.7241** |**26.53** |**0.8641** |15.0    |14.5
|[align-variant][assets]   |**37.33** |**0.9704** |1.7614     |**26.53** |0.8622     |**9.7** |**12.3**
|fuse-origin               |-         |-          |-          |-         |-          |8.0     |605.3
|[fuse-yolo][assets]       |-         |-          |-          |-         |-          |**4.4** |**74.8**

<small>\* The original model size exceeds github's release limitation (2GB). You are free to train a model with the provided commands.</small>

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
python3 train.py --data data/warpedcoco.yaml --hyp data/hyp.align.scratch.yaml --cfg models/align_yolo.yaml --weights weights/yolov5x.pt --batch-size 16 --img-size 128 --epochs 150 --adam --device 0 --mode align
mv runs/train/exp weights/align/warpedcoco
```

#### Step 2 (Alignment): Unsupervised finetuning on UDIS-D

```bash
python3 train.py --data data/udis.yaml --hyp data/hyp.align.finetune.udis.yaml --cfg models/align_yolo.yaml --weights weights/align/warpedcoco/weights/best.pt --batch-size 16 --img-size 128 --epochs 50 --adam --device 0 --mode align
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
python3 train.py --data data/udis.yaml --hyp data/hyp.fuse.scratch.yaml --cfg models/fuse_yolo.yaml --weights weights/yolov5m.pt --batch-size 4 --img-size 640 --epochs 30 --adam --device 0 --mode fuse --reg-mode crop
mv runs/train/exp weights/fuse/udis
```

#### Step 6 (Reconstruction): Generating the finally stitched results

```bash
python3 inference_fuse.py --weights weights/fuse/udis/weights/best.pt --source data/udis.yaml --task test --half --img-size 640 --reg-mode crop
```

## TODO
- [ ] **FP16 Compatibility**: FP16 data-type may cause strange values in division operations.
- [ ] **Fuse Optimization**: the hyp-params and  loss functions are not perfect right now.

## Comparison and Discussion

In order to improve the flexibility and speed up the training process, we made this reimplementation in pytorch. We also adjusted the networks, loss functions, data augmentation, and a considerable part of hyper-parameters. Taking the network as an example, in the alignment phase we replace the `CostVolume` module with a simple `Concat` module, of which the former is either time-consuming or memory-consuming, and even leads to divergence. And the alignment training process may be broken for some unknown reasons. This repository is far
away from perfect, and I hope you can assist me to complete this project. Contact me if you have any problems or suggestions -- kail@zju.edu.cn.
