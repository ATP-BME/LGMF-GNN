# MAMF-GCN
This is a Pytorch implementation of Multi-scale adaptive multi-channel fusion deep graph convolutional network for predicting mental disorder, as described in our paper.
# Requirement
Pytorch  

# Data
In order to use your own data, you have to provide  
an N by N adjacency matrix (N is the number of nodes),  
an N by D feature matrix (D is the number of features per node)，
Multi-scale features at different brain atlases were obtained by DPARSF

## Training
```
python train_eval_mamfgcn.py --train=1
```
If you want to train a new model on your own dataset, please change the data loader functions defined in `dataloader.py` accordingly, then run `python train_eval_mamfgcn.py --train=1`  

## Reference 
If you find this code useful in your work, please cite:
```
@article{pan2022mamf,
  title={MAMF-GCN: Multi-scale adaptive multi-channel fusion deep graph convolutional network for predicting mental disorder},
  author={Pan, Jiacheng and Lin, Haocai and Dong, Yihong and Wang, Yu and Ji, Yunxin},
  journal={Computers in Biology and Medicine},
  volume={148},
  pages={105823},
  year={2022},
  publisher={Elsevier}
}
```
