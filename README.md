# Unimodal-Weighted Label Distribution Learning (LDL)

This is a PyTorch implementation of the **Unimodal-Weighted Label Distribution Learning (LDL) approach**.  

<img width="866" height="266" alt="uw_ldl" src="https://github.com/user-attachments/assets/ef998bc3-8a48-497a-b521-ae2e3931e436" />


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
   
   datasets are includes in the "./data/" path.

   Five-label datasets:

   SST-5 datasets is recommended to be downloaded from https://nlp.stanford.edu/sentiment/. Here provides the version where we already preprocessed during the experiment.



### Training

See from the main.py code.

Note that The RoBERTa and BERT versions differ only in terms of model application. Both the models are on the huggingface and can be downloaded locally.
CNN model requires the additional glove-6B model. https://nlp.stanford.edu/projects/glove/

The run_grid_experiments function is used to do the grid search and can be expanded through the change of searching range.



---

## Comparison Methods
The major methods includes:

 Mean-Variance loss https://github.com/Herosan163/AgeEstimation

DLDL-v2 loss [gaobb/DLDL-v2: [IJCAI 2018\] Age Estimation Using Expectation of Label Distribution Learning](https://github.com/gaobb/DLDL-v2)



