# SimCLR

_A Simple Framework for Contrastive Learning of Visual Representations_

## Authors

Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton

## Abstract

This paper presents SimCLR: a simple framework for contrastive learning of visual representations. We simplify recently proposed contrastive self-supervised learning algorithms without requiring specialized architectures or a memory bank. In order to understand what enables the contrastive prediction tasks to learn useful representations, we systematically study the major components of our framework. We show that (1) composition of data augmentations plays a critical role in defining effective predictive tasks, (2) introducing a learnable nonlinear transformation between the representation and the contrastive loss substantially improves the quality of the learned representations, and (3) contrastive learning benefits from larger batch sizes and more training steps compared to supervised learning. By combining these findings, we are able to considerably outperform previous methods for self-supervised and semi-supervised learning on ImageNet. A linear classifier trained on self-supervised representations learned by SimCLR achieves 76.5% top-1 accuracy, which is a 7% relative improvement over previous state-of-the-art, matching the performance of a supervised ResNet-50. When fine-tuned on only 1% of the labels, we achieve 85.8% top-5 accuracy, outperforming AlexNet with 100X fewer labels.

[[paper]](https://arxiv.org/abs/2002.05709)

## Config file

Before running SimCLR, make sure you choose the correct running configurations on the ```config.yaml``` file.

```yaml

# A batch size of N, produces 2 * (N-1) negative samples. Original implementation uses a batch size of 8192
batch_size: 512

# Number of epochs to train
epochs: 40

# Frequency to eval the similarity score using the validation set
eval_every_n_epochs: 1

# Specify a folder containing a pre-trained model to fine-tune. If training from scratch, pass None.
fine_tune_from: 'resnet-18_80-epochs'

# Frequency to which tensorboard is updated
log_every_n_steps: 50

# l2 Weight decay magnitude, original implementation uses 10e-6
weight_decay: 10e-6

# if True, training is done using mixed precision. Apex needs to be installed in this case.
fp16_precision: False

# Model related parameters
model:
  # Output dimensionality of the embedding vector z. Original implementation uses 2048
  out_dim: 256 
  
  # The ConvNet base model. Choose one of: "resnet18" or "resnet50". Original implementation uses resnet50
  base_model: "resnet18"

# Dataset related parameters
dataset:
  s: 1
  
  # dataset input shape. For datasets containing images of different size, this defines the final 
  input_shape: (96,96,3)
  
  # Number of workers for the data loader
  num_workers: 0
  
  # Size of the validation set in percentage
  valid_size: 0.05

# NTXent loss related parameters
loss:
  # Temperature parameter for the contrastive objective
  temperature: 0.5
  
  # Distance metric for contrastive loss. If False, uses dot product. Original implementation uses cosine similarity.
  use_cosine_similarity: True
```

## Feature Evaluation

Feature evaluation is done using a linear model protocol.

Features are learned using the ```STL10 train+unsupervised``` set and evaluated in the ```test``` set;

|      Linear Classifier      | Feature Extractor | Architecture | Feature dimensionality | Projection Head  dimensionality | Epochs | STL10 Top 1 |
|:---------------------------:|:-----------------:|:------------:|:----------------------:|:-------------------------------:|:------:|:-----------:|
|     Logistic Regression     |    PCA Features   |       -      |           256          |                -                |        |    36.0%    |
|             KNN             |    PCA Features   |       -      |           256          |                -                |        |    31.8%    |
| Logistic Regression (LBFGS) |       SimCLR      |   ResNet-18  |           512          |               256               |   40   |    70.3%    |
|             KNN             |       SimCLR      |   ResNet-18  |           512          |               256               |   40   |    66.2%    |
| Logistic Regression (LBFGS) |       SimCLR      |   ResNet-18  |           512          |               256               |   80   |    72.9%    |
|             KNN             |       SimCLR      |   ResNet-18  |           512          |               256               |   80   |    69.8%    |
| Logistic Regression (Adam)  |       SimCLR      |   ResNet-18  |           512          |               256               |   100  |    75.4%    |
|  Logistic Regression (Adam) |       SimCLR      |   ResNet-50  |          2048          |               128               |   40   |    74.6%    |
|  Logistic Regression (Adam) |       SimCLR      |   ResNet-50  |          2048          |               128               |   80   |    77.3%    |

This repository provides the pretrained models for Logistic Regression, but not the pretrained models for KNN. So, if you want to use the KNN method, please train it by yourself.

You could find the provided pretrained models in [pretrained_models](./pretrained_models) directory.

## References

[1] Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton. [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)

[2] sthalles [SimCLR](https://github.com/sthalles/SimCLR)
