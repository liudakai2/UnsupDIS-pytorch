##### pre-train align model on WarpedCOCO from scratch
python3 train.py --data data/warpedcoco.yaml --hyp data/hyp.align.scratch.yaml --cfg models/align_yolo.yaml --weights weights/yolov5x.pt --batch-size 16 --img-size 128 --epochs 150 --adam --device 0 --mode align
## variant network with HalfInstanceNormalizarion
# python3 train.py --data data/warpedcoco.yaml --hyp data/hyp.align.scratch.yaml --cfg models/align_variant.yaml --weights '' --batch-size 16 --img-size 128 --epochs 150 --adam --device 0 --mode align
## original network, not recommended, it is too slow
# python3 train.py --data data/warpedcoco.yaml --hyp data/hyp.align.scratch.yaml --cfg models/align_origin.yaml --weights '' --batch-size 4 --img-size 128 --epochs 150 --adam --device 0 --mode align

##### finetune align model on UDIS-D
## `data/hyp.align.finetune.udis.yaml` is a very triky hyp-parameter set specified for UDIS-D. switch to `data/hyp.align.finetune.yaml` if necessary.
python3 train.py --data data/udis.yaml --hyp data/hyp.align.finetune.udis.yaml --cfg models/align_yolo.yaml --weights weights/align/coco/weights/best.pt --batch-size 16 --img-size 128 --epochs 50 --adam --device 0 --mode align
#python3 train.py --data data/udis.yaml --hyp data/hyp.align.finetune.udis.yaml --cfg models/align_variant.yaml --weights weights/align/coco/weights/best.pt --batch-size 16 --img-size 128 --epochs 50 --adam --device 0 --mode align
#python3 train.py --data data/udis.yaml --hyp data/hyp.align.finetune.udis.yaml --cfg models/align_origin.yaml --weights weights/align/coco/weights/best.pt --batch-size 4 --img-size 128 --epochs 50 --adam --device 0 --mode align

##### generate the coarsely aligned image pairs
python3 inference_align.py --weights weights/align/udis/weights/best.pt --source data/udis.yaml --task train
python3 inference_align.py --weights weights/align/udis/weights/best.pt --source data/udis.yaml --task test
mkdir UDIS-D/warp
mv runs/infer/exp UDIS-D/warp/train
mv runs/infer/exp2 UDIS-D/warp/test

##### train fuse model on UDIS-D
python3 train.py --data data/udis.yaml --hyp data/hyp.fuse.scratch.yaml --cfg models/fuse_yolo.yaml --weights weights/yolov5m.pt --batch-size 4 --img-size 640 --epochs 30 --adam --device 0 --mode fuse --reg-mode crop
## optional
# python3 train.py --data data/udis.yaml --hyp data/hyp.fuse.scratch.yaml --cfg models/fuse_yolo.yaml --weights weights/yolov5m.pt --batch-size 4 --img-size 512 --epochs 30 --adam --device 0 --mode fuse --reg-mode resize
# python3 train.py --data data/udis.yaml --hyp data/hyp.fuse.scratch.yaml --cfg models/fuse_origin.yaml --weights '' --batch-size 4 --img-size 512 --epochs 30 --adam --device 0 --mode fuse --reg-mode resize

##### generate the final fused results
python3 inference_fuse.py --weights weights/fuse/udis/weights/best.pt --source data/udis.yaml --task test --half --img-size 640 --reg-mode crop
# python3 inference_fuse.py --weights weights/fuse/udis/weights/best.pt --source data/udis.yaml --task test --half --img-size 512 --reg-mode resize