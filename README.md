# MVP-SegNet

## Table of contents
* [Introduction](#introduction)
* [Setting up](#setting-up-SPARC-SPy)
* [Reporting issues](#reporting-issues)
* [Cite us](#cite-us)

## Introduction
This project introduces MVP-SegNet, a deep learning model designed for accurate segmentation of stroke lesions in T1-weighted MRI scans. Stroke lesion segmentation is crucial for automating prognosis and recovery predictions, enabling more effective clinical interventions. 

Existing methods, including convolutional neural networks (CNNs) and transformers, often struggle to effectively capture both spatial and global features within medical images. Furthermore, many state-of-the-art models are computationally complex, hindering their deployment in real-world clinical settings. 

To address these challenges, MVP-SegNet incorporates several key innovations:

* **Multi-View Pyramidical Connections:** A novel residual connection technique that combines information from different receptive fields, enabling the model to capture contextual information at various levels of detail within the MRI images.
* **Squeeze-and-Excitation Blocks:** These blocks enhance the model's ability to learn more discriminative features by adaptively weighting the importance of different feature channels.

MVP-SegNet was rigorously evaluated on both the ATLAS and an in-house stroke dataset, demonstrating superior performance compared to baseline models such as U-Net and V-Net. Notably, MVP-SegNet achieved a significant 6% improvement in Dice score on the ATLAS v2.0 dataset, along with notable improvements in recall and mean Intersection over Union (IoU).

While recognizing certain limitations, this study demonstrates the potential of MVP-SegNet as a foundation for robust and clinically applicable stroke prediction models.


## Installation
### Pre-requisites 
- [Git](https://git-scm.com/)
- Python versions:
   - 3.9

### From source code
#### Downloading source code
Clone the MVP-SegNet repository from github, e.g.:
```
git clone https://github.com/MMal151/MVP-SegNet.git 
```

### Installing dependencies
```
pip install requirements.txt
```

## Reporting issues 
To report an issue or suggest a new feature, please use the [issues page](https://github.com/MMal151/MVP-SegNet/issues). 
Please check existing issues before submitting a new one.

### Project structure
* `/src/` - Directory of MVP-SegNet python module.
* `/Analysis/` - Directory containing scripts for statistical analysis.
* `config_train.yml` - Sample configuration file for training a new model.
* `config_inference.yml` - Sample configuration file for generating an inference (generate segmentations from test set).
* `config_dataset.yml` - Sample configuration file for pre-processing dataset.


## Cite us
If you use MVP-SegNet to make new discoveries or use the source code, please cite us as follows:
```
to-be-added 


