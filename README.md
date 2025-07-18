# 🧠 MindLLM: A Subject-Agnostic and Versatile Model for fMRI-to-Text Decoding

This is the official implementation of ICML 2025 paper [MindLLM: A Subject-Agnostic and Versatile Model for fMRI-to-Text Decoding](https://arxiv.org/abs/2502.15786).


## Installation
```
conda create -n mindllm python=3.10 -y
conda activate mindllm
pip install -r requirements.txt
pip install flash-attn==2.5.6 --no-build-isolation

# Java should be installed for some metrics
```
## Dataset
Make sure you have [git-lfs](https://git-lfs.com/) installed. The dataset is host on [Huggingface](https://huggingface.co/datasets/BoltzmachineQ/brain-instruction-tuning)

## Pretrained checkpoints
Model weights can be downloaded from [Huggingface](https://huggingface.co/BoltzmachineQ/MindLLM/).

## Usage

### Pretraining
Pretrain MindLLM on a single subject (e.g., subject 1)
```shell
python main.py group_by_coco=false
```
Only current-image-based datasets can use `group_by_coco=True`. As long as you include `coco-caption-previous`, this should be set to `False`.

Pretrain MindLLM on subjects 1-7
```shell
python main.py group_by_coco=false "subjects=[1,2,3,4,5,6,7]"
```

### Finetuning
To finetune on downstream tasks (e.g., results in Table 2). For example,
```shell
# COCO QA
python main.py data.task=coco-qa data.split_val=true early_stop=true checkpoint=/path/to/mindllm-base.ckpt lr=1e-4

# A-OKVQA
python main.py data.batch_size=4 data.task=a-okvqa early_stop=true data.split_val=true checkpoint=mindllm-base.ckpt lr=5e-4
```
