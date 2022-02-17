# Towards Open Vocabulary Object Detection without Human-provided Bounding Boxes

## Introduction
This is an official pytorch implementation of [Towards Open Vocabulary Object Detection without Human-provided Bounding Boxes](https://arxiv.org/pdf/2111.09452.pdf).
![network](figs/pipeline.jpg?raw=true)
## Environment
```angular2
UBUNTU="18.04"
CUDA="11.0"
CUDNN="8"
```

## Installation
```angular2
conda create --name ovd

conda activate ovd

cd $INSTALL_DIR

bash ovd_install.sh

git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

cd ../
cuda_dir="maskrcnn_benchmark/csrc/cuda"
perl -i -pe 's/AT_CHECK/TORCH_CHECK/' $cuda_dir/deform_pool_cuda.cu $cuda_dir/deform_conv_cuda.cu
python setup.py build develop
```
## Data Preparation
* Follow steps in [datasets/README.md](https://github.com/salesforce/PB-OVD/blob/master/datasets/README.md) for data preparation

## Inference
* Download our [pre-trained model](https://storage.cloud.google.com/sfr-pb-ovd-research/models/pretrain.pth) and [fine-tuned model](https://storage.cloud.google.com/sfr-pb-ovd-research/models/finetune.pth)

```angular2
python -m torch.distributed.launch --nproc_per_node=8 tools/test_net.py \
--config-file configs/eval.yaml \
MODEL.WEIGHT $PATH_TO_FINAL_MODEL \
OUTPUT_DIR $OUTPUT_DIR
```
* For LVIS, use their official API to get evaluated numbers

```angular2
python evaluate_lvis_official.py --coco_anno_path datasets/lvis_v0.5_val_all_clipemb.json \
--result_dir $OUTPUT_DIR/inference/lvis_v0.5_val_all_cocostyle/
```
## Pretrain with Pseudo Labels

```angular2
python -m torch.distributed.launch --nproc_per_node=16 tools/train_net.py  --distributed \
--config-file configs/pretrain_1m.yaml \
OUTPUT_DIR $OUTPUT_DIR
```

## Finetune

```angular2
python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py  --distributed \
--config-file configs/finetune.yaml \
MODEL.WEIGHT $PATH_TO_PRETRAIN_MODEL \
OUTPUT_DIR $OUTPUT_DIR
```

## Generate Your Own Pseudo Box Labels
![examples](figs/examples.jpg?raw=true)

### Installation

```angular2
conda create --name gen_plabels

conda activate gen_plabels

bash gen_plabel_install.sh
```
### Preparation

* Referring [examples/README.md](https://github.com/salesforce/PB-OVD/blob/master/examples/README.md) for data preparation

### Generate Pseudo Labels
* Get pseudo labels based on [ALBEF](https://arxiv.org/abs/2107.07651)

```angular2
python pseudo_bbox_generation.py
```

* Organize dataset in COCO format
```angular2
python prepare_coco_dataset.py
```

* Extract text embedding using [CLIP](https://arxiv.org/abs/2103.00020)

```angular2
# pip install git+https://github.com/openai/CLIP.git

python prepare_clip_embedding_for_open_vocab.py
```

* Check your final pseudo labels by visualization

```angular2
python visualize_coco_style_dataset.py
```

## Citation
* If you find this code helpful, please cite our paper:
``` latex
@article{gao2021towards,
  title={Towards Open Vocabulary Object Detection without Human-provided Bounding Boxes},
  author={Gao, Mingfei and Xing, Chen and Niebles, Juan Carlos and Li, Junnan and Xu, Ran and Liu, Wenhao and Xiong, Caiming},
  journal={arXiv preprint arXiv:2111.09452},
  year={2021}
}
```

## Contact

* Please send an email to mingfei.gao@salesforce.com or cxing@salesforce.com if you have questions.

## Notes

* Files obtained from [maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) are covered under the MIT license.