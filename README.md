# LGMF-GNN
Code of paper "An Objective Quantitative Diagnosis of Depression Using a Local-to-global Multi-modal Fusion Graph Neural Network"

## Overview
Content includes source code for the implementation of local-to-global multi-modal fusion graph neural network on large multi-site MDD fMRI datasets for major depressive disorder diagnosis.

## Requirements
- python >= 3.8
- torch = 1.13.0
- torch-cluster = 1.6.1+pt113cu117
- torch-geometric = 2.3.1
- torch-scatter = 2.1.1+pt113cu117
- torch-sparse = 0.6.17+pt113cu117

## datasets
- [SRPBS dataset](https://doi.org/10.7303/syn22317081)
- [REST-meta-MDD dataset](https://doi.org/10.57760/sciencedb.o00115.00013)
- [OpenNeuro dataset](https://openneuro.org/datasets/ds002748)
- [Anding dataset](https://pan.baidu.com/s/15pjQske5yGaiyz-7_TEVqg)

## Toolboxes
- [DPARSF](http://www.rfmri.org/DPARSF)
- [NeuroCombat-sklearn](https://github.com/Warvito/neurocombat_sklearn)

## Citation
```
@article{LIU2024101081,
title = {An objective quantitative diagnosis of depression using a local-to-global multimodal fusion graph neural network},
journal = {Patterns},
volume = {5},
number = {12},
pages = {101081},
year = {2024},
issn = {2666-3899},
doi = {https://doi.org/10.1016/j.patter.2024.101081},
url = {https://www.sciencedirect.com/science/article/pii/S266638992400240X},
author = {Shuyu Liu and Jingjing Zhou and Xuequan Zhu and Ya Zhang and Xinzhu Zhou and Shaoting Zhang and Zhi Yang and Ziji Wang and Ruoxi Wang and Yizhe Yuan and Xin Fang and Xiongying Chen and Yanfeng Wang and Ling Zhang and Gang Wang and Cheng Jin},
keywords = {major depressive disorder, multimodal fusion, graph neural network, brain connectivity analysis, neuroimaging biomarkers},
abstract = {Summary
This study developed an artificial intelligence (AI) system using a local-global multimodal fusion graph neural network (LGMF-GNN) to address the challenge of diagnosing major depressive disorder (MDD), a complex disease influenced by social, psychological, and biological factors. Utilizing functional MRI, structural MRI, and electronic health records, the system offers an objective diagnostic method by integrating individual brain regions and population data. Tested across cohorts from China, Japan, and Russia with 1,182 healthy controls and 1,260 MDD patients from 24 institutions, it achieved a classification accuracy of 78.75%, an area under the receiver operating characteristic curve (AUROC) of 80.64%, and correctly identified MDD subtypes. The system further discovered distinct brain connectivity patterns in MDD, including reduced functional connectivity between the left gyrus rectus and right cerebellar lobule VIIB, and increased connectivity between the left Rolandic operculum and right hippocampus. Anatomically, MDD is associated with thickness changes of the gray and white matter interface, indicating potential neuropathological conditions or brain injuries.}
}
```
