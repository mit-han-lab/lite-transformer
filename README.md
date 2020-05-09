# Lite Transformer with Long-Short Range Attention
```
@inproceedings{Wu2020LiteTransformer,
  title={Lite Transformer with Long-Short Range Attention},
  author={Zhanghao Wu* and Zhijian Liu* and Ji Lin and Yujun Lin and Song Han},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020}
}
```

## Overview
We release the PyTorch code for the Lite Transformer. [[Paper](https://arxiv.org/abs/2004.11886v1)|[Website](https://zhanghaowu.me/pubs/LiteTransformer/index.html)|[Slides](https://zhanghaowu.me/assets/pdf/Presentation_LiteTransformer.pdf)]:
![overview](figures/overview.png?raw=true "overview")

### Consistent Improvement by Tradeoff Curves
![tradeoff](figures/tradeoff.png?raw=true "tradeoff")
### Save 20000x Searching Cost of Evolved Transformer
![et](figures/et.png?raw=true "et")
### Further Compress Transformer by 18.2x
![compression](figures/compression.png?raw=true "compression")

## How to Use

### Prerequisite

* Python version >= 3.6
* [PyTorch](http://pytorch.org/) version >= 1.0.0
* configargparse >= 0.14
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

### Installation

1. Codebase
    
    To install fairseq from source and develop locally:
    ```bash
    pip install --editable .
    ```

2. Costumized Modules

    We also need to build the `lightconv` and `dynamicconv` for GPU support.

    Lightconv_layer
    ```bash
    cd fairseq/modules/lightconv_layer
    python cuda_function_gen.py
    python setup.py install
    ```
    Dynamicconv_layer
    ```bash
    cd fairseq/modules/dynamicconv_layer
    python cuda_function_gen.py
    python setup.py install
    ```

### Data Preparation
#### IWSLT'14 De-En
We follow the data preparation in [fairseq](github.com/pytorch/fairseq). To download and preprocess the data, one can run
```bash
bash configs/iwslt14.de-en/prepare.sh
```

#### WMT'14 En-Fr
We follow the data pre-processing in [fairseq](github.com/pytorch/fairseq).  To download and preprocess the data, one can run
```bash
bash configs/wmt14.en-fr/prepare.sh
```

#### WMT'16 En-De
We follow the data pre-processing in [fairseq](github.com/pytorch/fairseq). One should first download the preprocessed data from the [Google Drive](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) provided by Google. To binarized the data, one can run
```bash
bash configs/wmt16.en-de/prepare.sh [path to the downloaded zip file]
```

### Testing

For example, to test the models on WMT'14 En-Fr, one can run
```bash
configs/wmt14.en-fr/test.sh [path to the model checkpoints] [gpu-id] [test|valid]
```
For instance, to evaluate Lite Transformer on GPU 0 (with the BLEU score on test set of WMT'14 En-Fr), one can run
```bash
configs/wmt14.en-fr/test.sh en-fr-496.pt 0 test
```

### Training
We provided several examples to train Lite Transformer with this repo:

To train Lite Transformer on WMT'14 En-Fr (with 8 GPUs), one can run
```bash
python train.py data/binary/wmt14_en_fr --configs configs/wmt14.en-fr/attention/multibranch_v2/embed496.yml
```
To train Lite Transformer with less GPUs, e.g. 4 GPUS, one can run
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py data/binary/wmt14_en_fr --configs configs/wmt14.en-fr/attention/multibranch_v2/embed496.yml --update-freq 32
```
In general, to train a model, one can run
```bash
python train.py [path to the data binary] --configs [path to config file] [override options]
```
Note that `--update-freq` should be adjusted according to the GPU numbers (16 for 8 GPUs, 32 for 4 GPUs).

### Distributed Training (optional)

To train Lite Transformer in distributed manner. For example on two GPU nodes with totally 16 GPUs.
```bash
# On host1
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --nnodes=2 --node_rank=0 \
        --master_addr=host1 --master_port=8080 \
        train.py data/binary/wmt14_en_fr \
        --configs configs/wmt14.en-fr/attention/multibranch_v2/embed496.yml \
        --distributed-no-spawn \
        --update-freq 8
# On host2
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --nnodes=2 --node_rank=1 \
        --master_addr=host1 --master_port=8080 \
        train.py data/binary/wmt14_en_fr \
        --configs configs/wmt14.en-fr/attention/multibranch_v2/embed496.yml \
        --distributed-no-spawn \
        --update-freq 8
```

## Models
We provide the checkpoints for our Lite Transformer reported in the paper:
| Dataset | \#Mult-Adds | Test Score | Model and Test Set |
|:--:|:--:|:--:|:--:|
| [WMT'14 En-Fr](http://statmt.org/wmt14/translation-task.html#Download) | 90M | 22.5 |[download](https://drive.google.com/open?id=10Iotg0dnt9sJTqEghtNhIIwJL1R3LYBe) |
| | 360M | 25.6 | [download](https://drive.google.com/open?id=10WMpIrdnDRWa_7afYJsqiiONdWlTLrJs) |
| | 527M | 26.5 | [download](https://drive.google.com/open?id=10Wfv80wOTkL-hkXNyxM8IVlcroHuuUvA) |
| [WMT'16 En-De](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) | 90M | 35.3 | [download](https://drive.google.com/open?id=10ArxzUsMZ8gDe6zw5d3xTHYmeUasys1q) |
| | 360M | 39.1 | [download](https://drive.google.com/open?id=10Fd1iXFiOtuwjxm1K8S2RqiEeCuDhxYn) |
| | 527M | 39.6 | [download](https://drive.google.com/open?id=10HYj-rcJ4CIPp-BtpckkmYIgzH5Urrz0)|
| [CNN / DailyMail](https://github.com/abisee/cnn-dailymail) | 800M | 38.3 (R-L) | [download](https://drive.google.com/open?id=14sQZ_H7HMQGhL7Ko1WkktWUvbEslOeu9)|

