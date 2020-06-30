# Divide-and-Rule: Self-Supervised Learning for
Survival Analysis in Colorectal Cance

With the long-term rapid increase in incidences of colorectal
cancer (CRC), there is an urgent clinical need to improve risk strati-
fication. The conventional pathology report is usually limited to only
a few histopathological features. However, most of the tumor microen-
vironments used to describe patterns of aggressive tumor behavior are
ignored. In this work, we aim to learn histopathological patterns within
cancerous tissue regions that can be used to improve prognostic strat-
ification for colorectal cancer. To do so, we propose a self-supervised
learning method that jointly learns a representation of tissue regions as
well as a metric of the clustering to obtain their underlying patterns.
These histopathological patterns are then used to represent the interac-
tion between complex tissues and predict clinical outcomes directly. We
furthermore show that the proposed approach can benefit from linear
predictors to avoid overfitting in patient outcomes predictions. To this
end, we introduce a new well-characterized clinicopathological dataset,
including a retrospective collective of 374 patients, with their survival
time and treatment information. Histomorphological clusters obtained
by our method are evaluated by training survival models. The experi-
mental results demonstrate statistically significant patient stratification
and our approach outperformed state-of-the-art deep clustering methods.

## Requirements
* pytorch = 1.2.0
* torchvision = 0.4.0
* numpy = 1.17

## Setup

The pre-trained models are available on the google drive [link](https://drive.google.com/drive/folders/1Veb-3STH74GKCr-AyhKQRnEHa743P6Ff?usp=sharing). We provide as well a small dataset to try the model by yourself.
You can run the training using the command

```bash
python run_dnr.py --db sample.hdf5 --pretrained dnr_model_state --output .
```

