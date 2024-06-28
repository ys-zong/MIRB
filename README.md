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

## Results

| **Models**              | **Reasoning** | **Knowledge** | **Perception** | **Multi-Hop** | **Average** |
|-------------------------|---------------|---------------|----------------|---------------|-------------|
| Random Chance           | 20.80         | 37.62         | 21.42          | 0.00          | 23.02       |
| **LLaVA-v1.5-7B**       | 48.86         | 27.14         | 37.89          | 0.00          | 28.47       |
| **LLaVA-Next-7B**       | 48.40         | 29.35         | 41.56          | 0.00          | 29.83       |
| **LLaVA-Next-13B**      | 48.44         | 29.85         | 40.22          | 0.00          | 29.38       |
| **Qwen-VL-Chat**        | 19.23         | 13.87         | 24.44          | 0.00          | 14.38       |
| **InternLM-XComposer2** | 54.74         | _37.23_       | 37.22          | 0.81          | 32.50       |
| **VILA-2.7B**           | 53.27         | 31.01         | 48.33        | 0.00          | 33.15       |
| **VILA-7B**             | 63.66       | 35.31         | 47.11          | 0.00          | 36.52     |
| **Emu2-Chat**           | 40.40         | 24.51         | 44.00          | 0.00          | 27.23       |
| **IDEFICS1-9B**         | 45.89         | 23.49         | 36.89          | 0.00          | 26.57       |
| **IDEFICS2-8B**         | 61.26         | 31.83         | 39.00          | 0.00          | 33.02       |
| **Mantis-IDEFICS2**     | 58.73         | 33.78         | 46.78          | 0.00          | 34.82       |
| **LongVA-7B**           | _66.63_         | 35.31         | _48.89_          | 0.00          | _37.71_       |
| **Phi-3-Vision**        | 60.19         | 34.49         | 46.22          | 0.00          | 35.23       |
| **GPT-4V**              | **75.66**     | **50.59**     | **49.67**      | **36.29**     | **53.05**   |


## Citations

```
@article{zhao2024mirb
  author    = {Bingchen Zhao, Yongshuo Zong, Letian Zhang, Timothy Hospedales},
  title     = {Benchmarking Multi-Image Understanding in Vision and Language Models: Perception, Knowledge, Reasoning, and Multi-Hop Reasoning},
  journal   = {arXiv preprint},
  year      = {2024},
}
```
