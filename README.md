### DualCon

Code for [Dual-Level Contrastive Alignment for Robust Representations in Long-Tailed Medical VQA]

### Environment Setup and Package Installation
```
conda create -n DualCon python=3.9.6 -y  
conda activate DualCon  
conda install pytorch==2.5.1 torchvision==0.15.2 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia  
pip install -r requirements.txt  
```
### Create directories

Before training and evaluating the model, create a folder named checkpoint. This directory will be used to store the trained models and prediction results.

### Download datasets

* [VQA-RAD](https://osf.io/89kps/)
* [SLAKE](https://www.med-vqa.com/slake/)
* [Path-VQA](https://pathvqachallenge.grand-challenge.org/)

To obtain the PMC-600K dataset used for pretraining, execute the following commands:
```
wget https://github.com/microsoft/LLaVA-Med/blob/main/data/llava_med_image_urls.jsonl  
mkdir pmc_articles  
mkdir images  
python script.py  
```
### Train and Evaluate the Model
run:
```
python run.py
```

The code is based on [LLaVA-Med](https://github.com/microsoft/LLaVA-Med). We thank the authors for their open-sourced code.
