# Unimodal-Weighted Label Distribution Learning (LDL)

This is a PyTorch implementation of the **Unimodal-Weighted Label Distribution Learning (LDL) approach**.  



---

## Dependencies
- Python 3.10
- PyTorch 2.7.1

Tested on:
- CUDA 12.6
- GPU NVIDIA A800-PCIE-80GB

---

## Usage
### Data Preparation
1. The three-label datasets are processed differently from the five-label datasets.

2. dataset links: 

   Three-label datasets: 

   https://huggingface.co/FinanceInc/datasets

   https://huggingface.co/zeroshot/datasets

   Five-label datasets:

   https://nlp.stanford.edu/sentiment/

   The datasets are placed in "./data/" path.



### Training

See from the train.py code.

Note that The RoBERTa and BERT versions differ only in terms of model application.

The models are loaded by local path.



---

## Comparison Methods
The major ethods includes:

 mean-variance loss https://github.com/Herosan163/AgeEstimation

DLDL-v2 loss [gaobb/DLDL-v2: [IJCAI 2018\] Age Estimation Using Expectation of Label Distribution Learning](https://github.com/gaobb/DLDL-v2)



