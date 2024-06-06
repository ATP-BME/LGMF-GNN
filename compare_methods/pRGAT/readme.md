Graph CNNs for population graphs: classification of the ABIDE dataset
---------------------------------------------------------------------

#### INSTALLATION  

The root folder in fetch_data.py (line 12) and ABIDEParser.py (line 17) has to be updated to the folder were the data will be stored. 

Next, to install, organise and pre-process the ABIDE database:
python fetch_data.py


#### USAGE

To run the programme with default parameters: 
 ```python
python main_ABIDE_F_F_GAT.py
python main_ABIDE_F_RGAT.py
python main_ABIDE_PF_F_GAT.py
python main_ABIDE_FP_RGAT.py 
```
 
To get a detailed description of parameters:
 ```python
python main_ABIDE_F_F_GAT.py --help
python main_ABIDE_F_RGAT.py --help
python main_ABIDE_PF_F_GAT.py --help
python main_ABIDE_FP_RGAT.py --help
 ```


#### REQUIREMENTS 

pytorch 1.4.0 <br />
torchvision 0.5.0 <br />
networkx <br />
nilearn <br />
scikit-learn <br />
joblib <br />
torch-cluster 1.5.4  <br />                                  
torch-scatter 2.0.4  <br />                 
torch-sparse  0.6.1  <br />                 
torch-spline-conv 1.2.0  <br />      
torch-geometric 1.7.2  <br />            





