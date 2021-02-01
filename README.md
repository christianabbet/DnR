# Divide-and-Rule: Self-Supervised Learning for Survival Analysis in Colorectal Cancer

Pytorch implementation of Divide-and-Rule ([DnR](https://arxiv.org/abs/2007.03292)) paper.

With the long-term rapid increase in incidences of colorectal cancer (CRC), there is an urgent clinical need to improve 
risk stratification. The conventional pathology report is usually limited to only a few histopathological features. 
However, most of the tumor microenvironments used to describe patterns of aggressive tumor behavior are ignored. In
this work, we aim to learn histopathological patterns within cancerous tissue regions that can be used to improve
prognostic stratification for colorectal cancer. To do so, we propose a self-supervised learning method that jointly 
learns a representation of tissue regions as well as a metric of the clustering to obtain their underlying patterns.
These histopathological patterns are then used to represent the interaction between complex tissues and predict 
clinical outcomes directly. We furthermore show that the proposed approach can benefit from linear predictors to 
avoid overfitting in patient outcomes predictions. To this end, we introduce a new well-characterized 
clinicopathological dataset, including a retrospective collective of 374 patients, with their survival time and 
treatment information. Histomorphological clusters obtained by our method are evaluated by training survival models. 
The experimental results demonstrate statistically significant patient stratification and our approach outperformed 
state-of-the-art deep clustering methods.

![Pipeline full](figs/sketch_pipeline_full.png)

## Requirements
* pytorch = 1.2.0
* torchvision = 0.4.0
* numpy = 1.17

## Setup

The pre-trained models are available on the google drive 
[link](https://drive.google.com/drive/folders/1Veb-3STH74GKCr-AyhKQRnEHa743P6Ff?usp=sharing). 
We provide as well a small dataset to try the model by yourself.
You can run the training using the command

```bash
python run_dnr.py --db sample.npy --pretrained dnr_model_state --output .
```

Here is an overview of the provided data as `samples.npy`.
```
samples.npy
|
├── [0]                  # Random location within slides (coordinates and slides not provided)
|   ├── image            # (224x224x3) RGB crop of WSI at the consdered location
|   ├── image_he:        # (224x224x2) H&E version of the "image" crop    
|   ├── image_pairs:     # (224x224x3) RGB crop overlapping with "image" crop
|   ├── image_pairs_he:  # (224x224x2) H&E version of the "image_pairs_he" crop 
|   └── idx_overall:     # (int) Used intervally when developping the alogithm - Not used
|
├── [1]                  # Another location
|   └── ...
|
└── [2]                  # Another location
|   └── ...
...

```

## Cite 
If you use this work please use the following citation :).
```text
@inproceedings{abbet2020divide,
  title={Divide-and-Rule: Self-Supervised Learning for Survival Analysis in Colorectal Cancer},
  author={Abbet, Christian and Zlobec, Inti and Bozorgtabar, Behzad and Thiran, Jean-Philippe},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={480--489},
  year={2020},
  organization={Springer}
}
```