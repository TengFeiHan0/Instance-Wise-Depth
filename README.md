# Instance-wise Depth Estimation (under construction)
In this work,we are aiming at predicting instance-wise depth via self-supervised learning mechanism. The introduction of instance segmentation network is essential to dealing with dynamic objects, in turn a high precise depth map is also helpful for object detection/instance segmentation tasks. 

* [monodepth2](https://github.com/nianticlabs/monodepth2)
* [centermask_plus](https://github.com/TengFeiHan0/CenterMask_plus)


## AdelaiDet

generally speaking, any instance seg networks could be inserted into our system, however, here is an open source toolbox including a series of instance segmentation algorithms. Up to now, These algorithms have achieved better performance than most SOTA methods on COCO dataset.

AdelaiDet consists of the following algorithms:

* [FCOS](https://github.com/tianzhi0549/FCOS)
* [BlendMask](https://arxiv.org/abs/2001.00309) _to be released_
* [SOLO](https://arxiv.org/abs/1912.04488) _to be released_

## Note
Initialy, I was planning to develop my own system based on the mentioned open-source framework. However, releasing their implementations may need a longer time than we expect. Therefore, I resort to [CenterMask](https://github.com/youngwanLEE/CenterMask), which is also a one-stage instance sgementation network based on FCOS. The original implementation is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark), but I reformulate their code, add some new features,  and will be opened at [here](https://github.com/TengFeiHan0/CenterMask_plus). 

## Installation

It should be mentioned that AdelaiDet is extended from Detectron2, so please install Detectron2 following the official guide: [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). Then build AdelaiDet with:
```
git clone https://github.com/aim-uofa/adet.git
cd adet
python setup.py build develop
```

## Quick Start

### Inference with Pre-trained Models

1. Pick a model and its config file, for example, `fcos_R_50_1x.yaml`.
2. Download the model `wget https://cloudstor.aarnet.edu.au/plus/s/glqFc13cCoEyHYy/download -O fcos_R_50_1x.pth`
3. Run the demo with
```
python demo/demo.py \
    --config-file configs/FCOS-Detection/R_50_1x.yaml \
    --input input1.jpg input2.jpg \
	--opts MODEL.WEIGHTS fcos_R_50_1x.pth
```

### Train Your Own Models

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md),
then run:

```
python tools/train_net.py \
    --config-file configs/FCOS-Detection/R_50_1x.yaml \
    --num-gpus 8 \
    OUTPUT_DIR training_dir/fcos_R_50_1x
```

The configs are made for 8-GPU training. To train on another number of GPUs, change the `num-gpus`.

