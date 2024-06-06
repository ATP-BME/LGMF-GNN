# SFGL
The source code for **s**pecificity-aware **f**ederated **g**raph **l**earning (SFGL) framework.

## Paper
**Preserving Specificity in Federated Graph Learning for fMRI-based Neurological Disorder Identification**

Junhao Zhang, Qianqian Wang, Xiaochuan Wang, Lishan Qiao, Mingxia Liu

## Datasets
We used the following datasets:

- ABIDE (Can be downloaded [here](http://fcon_1000.projects.nitrc.org/indi/abide/))
- REST-meta-MDD (Can be downloaded [here](http://rfmri.org/REST-meta-MDD))

## Dependencies
SFGL needs the following dependencies:

- python 3.8.10
- torch == 1.9.0
- numpy == 1.21.1
- einops == 0.6.1
- scipy == 1.7.1
- sklearn == 0.0
- tqdm == 4.63.0
- pandas == 1.3.2

## Structure
    - `./training.py`: The main functions for SFGL.
    - `./abide_dataset.py`: Data preparation for ABIDE.
    - `./mdd_dataset.py`: Data preparation for REST-meta-MDD.
    - `./model.py`: The model used in SFGL.
    - `./weight_avg.py`: This is federal aggregation function.
    - `./bold.py`: This is used to construct the dynamic FCN.
    - `./option.py`: This is used to adjust the options.
