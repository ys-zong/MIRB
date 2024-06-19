# Benchmarking Multi-Image Understanding in Vision and Language Models: Perception, Knowledge, Reasoning, and Multi-Hop Reasoning

[[Paper]](https://arxiv.org/abs/2406.12742)[[Dataset]](https://huggingface.co/datasets/VLLMs/MIRB)[[Code]](https://github.com/ys-zong/MIRB)


<div align="center">
<img src=assets/teaser_mirb.jpg width=60% />
</div>

## Abstract
> The advancement of large language models (LLMs) has significantly broadened the scope of applications in natural language processing, with multi-modal LLMs extending these capabilities to integrate and interpret visual data. 
> However, existing benchmarks for visual language models (VLMs) predominantly focus on single-image inputs, neglecting the crucial aspect of multi-image understanding. 
> In this paper, we introduce a Multi-Image Relational Benchmark MIRB,  designed to evaluate VLMs' ability to compare, analyze, and reason across multiple images. 
> Our benchmark encompasses four categories: perception, visual world knowledge, reasoning, and multi-hop reasoning. 
> Through a comprehensive evaluation of a wide range of open-source and closed-source models, we demonstrate that while open-source VLMs were shown to approach the performance of GPT-4V in single-image tasks, a significant performance gap remains in multi-image reasoning tasks. 
> Our findings also reveal that even the state-of-the-art GPT-4V model struggles with our benchmark, underscoring the need for further research and development in this area. 
> We believe our contribution of MIRB could serve as a testbed for developing the next-generation multi-modal models. 


![](https://github.com/DTennant/MIRB_eval/blob/main/assets/Data_samples.jpg?raw=true)

## Environment
```bash
conda create -n MIRB python==3.10 -y
conda activate MIRB
pip install -r requirements.txt
# optional
# pip install flash-attn --no-build-isolation --no-cache-dir
```

You should be able to run most of the models now, but may also want to check some models for specific requirements such as [LLaVA](https://github.com/haotian-liu/LLaVA), [VILA](https://github.com/Efficient-Large-Model/VILA), and [Qwen-VL](https://github.com/QwenLM/Qwen-VL).

## Data
Put huggingface data in `./MIR` and unzip `./MIR/images.zip`.

## Inference

Quick Start:

```bash
python inference.py --engine phi3-vision idefics2-8b --dataset codeu analogy
```
Results will be saved in `results` folder.

## Evaluation
```bash
python evaluate.py --engine phi3-vision idefics2-8b --dataset codeu analogy

```

## Citations

```
@article{zhao2024mirb
  author    = {Bingchen Zhao, Yongshuo Zong, Letian Zhang, Timothy Hospedales},
  title     = {Benchmarking Multi-Image Understanding in Vision and Language Models: Perception, Knowledge, Reasoning, and Multi-Hop Reasoning},
  journal   = {arXiv preprint},
  year      = {2024},
}
```
