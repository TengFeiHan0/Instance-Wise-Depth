# Instance-wise Depth Estimation (constructing)
In this work,we are aiming at predicting instance-wise depth via self-supervised learning mechanism. The orginal depth estimation module is originated from [monodepth2](https://github.com/nianticlabs/monodepth2), while any one stage off-the-shelf instance segmentation network could be inserted into our system. 


## AdelaiDet

an open source toolbox including a series of instance segmentation algorithms, I develop my own system.

AdelaiDet consists of the following algorithms:

* [FCOS]
* [BlendMask](https://arxiv.org/abs/2001.00309) _to be released_
* [SOLO](https://arxiv.org/abs/1912.04488) _to be released_


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


## Citing AdelaiDet

If you use this toolbox in your research or wish to refer to the baseline results, please use the following BibTeX entries.

```BibTeX
@inproceedings{tian2019fcos,
  title     =  {{FCOS}: Fully Convolutional One-Stage Object Detection},
  author    =  {Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle =  {Proc. Int. Conf. Computer Vision (ICCV)},
  year      =  {2019}
}

@article{chen2020blendmask,
  title   =  {BlendMask: Top-Down Meets Bottom-Up for Instance Segmentation},
  author  =  {Chen, Hao and Sun, Kunyang and Tian, Zhi and Shen, Chunhua and Huang, Yongming and Yan, Youliang},
  journal =  {arXiv preprint arXiv:2001.00309},
  year    =  {2020}
}

@article{wang2019solo,
  title   =  {SOLO: Segmenting Objects by Locations},
  author  =  {Wang, Xinlong and Kong, Tao and Shen, Chunhua and Jiang, Yuning and Li, Lei},
  journal =  {arXiv preprint arXiv:1912.04488},
  year    =  {2019}
}

@article{tian2019directpose,
  title   =  {{DirectPose}: Direct End-to-End Multi-Person Pose Estimation},
  author  =  {Tian, Zhi and Chen, Hao and Shen, Chunhua},
  journal =  {arXiv preprint arXiv:1911.07451},
  year    =  {2019}
}
```

## License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact the authors.
