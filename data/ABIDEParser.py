
import os
import csv
import numpy as np
import scipy.io as sio
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE
from nilearn import connectome
from scipy.spatial import distance
import SimpleITK as sitk

# Reading and computing the input data

# Selected pipeline
pipeline = 'cpac'

# Input data variables
#root_folder = '/bigdata/fMRI/ABIDE/'
# root_folder = '/media/pjc/expriment/mdd_exam/pjc/DATA/ABIEDE_AAL/'
root_folder = "/data0/lsy/SRPBS_new"
root_folder2 = "/data0/lsy/SRPBS_new"

data_folder = os.path.join(root_folder, 'ROISignal_AAL')
# phenotype = os.path.join(root_folder, 'participants.csv')
phenotype = "D:\data\depression\REST_meta_MDD\SRPBS_MDD_pheno.csv"
data_folder2 = os.path.join(root_folder2,'ROI_signal_extract_cc200')

def fetch_filenames(subject_IDs, file_type):

    """
        subject_list : list of short subject IDs in string format
        file_type    : must be one of the available file types

    returns:

        filenames    : list of filetypes (same length as subject_list)
    """

    import glob

    # Specify file mappings for the possible file types
    filemapping = {'func_preproc': '_func_preproc.nii.gz',
                   'rois_aal': '_rois_aal.1D'}

    # The list to be filled
    filenames = []

    # Fill list with requested file paths
    for i in range(len(subject_IDs)):
        os.chdir(data_folder)  # os.path.join(data_folder, subject_IDs[i]))   os.chdir() 方法用于改变当前工作目录到指定的路径
        try:
            filenames.append(glob.glob('*' + subject_IDs[i] + filemapping[file_type])[0])
        except IndexError:
            # Return N/A if subject ID is not found
            filenames.append('N/A')

    return filenames


# Get timeseries arrays for list of subjects
def get_timeseries(subject_list, atlas_name):
    """
        subject_list : list of short subject IDs in string format
        atlas_name   : the atlas based on which the timeseries are generated e.g. aal, cc200

    returns:
        time_series  : list of timeseries arrays, each of shape (timepoints x regions)
    """

    timeseries = []
    for i in range(len(subject_list)):
        subject_folder = os.path.join(data_folder, subject_list[i])
        ro_file = [f for f in os.listdir(subject_folder) if f.endswith('_rois_' + atlas_name + '.1D')]
        fl = os.path.join(subject_folder, ro_file[0])
        print("Reading timeseries file %s" %fl)
        timeseries.append(np.loadtxt(fl, skiprows=0))

    return timeseries


# Compute connectivity matrices
def subject_connectivity(timeseries, subject, atlas_name, kind, save=True, save_path=data_folder):
    """
        timeseries   : timeseries table for subject (timepoints x regions)
        subject      : the subject ID
        atlas_name   : name of the parcellation atlas used
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        save         : save the connectivity matrix to a file
        save_path    : specify path to save the matrix if different from subject folder

    returns:
        connectivity : connectivity matrix (regions x regions)
    """

    print("Estimating %s matrix for subject %s" % (kind, subject))

    if kind in ['tangent', 'partial correlation', 'correlation']:
        conn_measure = connectome.ConnectivityMeasure(kind=kind)
        connectivity = conn_measure.fit_transform([timeseries])[0]

    if save:
        subject_file = os.path.join(save_path, subject,
                                    subject + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat')
        sio.savemat(subject_file, {'connectivity': connectivity})

    return connectivity


# Get the list of subject IDs
def get_ids(num_subjects=None):
    """

    return:
        subject_IDs    : list of all subject IDs
    """

    # subject_IDs = np.genfromtxt(os.path.join(data_folder, 'same_protocol_train.txt'), dtype=str)
    # subject_IDs = np.genfromtxt(os.path.join(data_folder, 'same_protocol_T1.txt'), dtype=str)
    # subject_IDs = np.genfromtxt(os.path.join(data_folder, 'same_protocol_balanced.txt'), dtype=str)
    subject_IDs = np.genfromtxt(os.path.join("/data0/lsy/sub_list/same_protocol_balanced.txt"), dtype=str)


    

    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]

    return subject_IDs


# Get phenotype values for a list of subjects
def get_subject_score(subject_list, score):
    scores_dict = {}

    with open(phenotype,mode="r", encoding='utf-8-sig') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['participant_id'] in subject_list:
                scores_dict[row['participant_id']] = row[score]
    # print(subject_list)
    return scores_dict


# Dimensionality reduction step for the feature vector using a ridge classifier
def feature_selection(features, labels, train_ind, fnum):
    """
        matrix       : feature matrix (num_subjects x num_features)
        labels       : ground truth labels (num_subjects x 1)
        train_ind    : indices of the training samples
        fnum         : size of the feature vector after feature selection

    return:
        x_data      : feature matrix of lower dimension (num_subjects x fnum)
    """

    estimator = RidgeClassifier()
    selector = RFE(estimator, n_features_to_select=fnum, step=100, verbose=0)
    # 岭回归后按照重要性大小排序

    featureX = features[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(features)

    return x_data


# Make sure each site is represented in the training set when selecting a subset of the training set
def site_percentage(train_ind, perc, subject_list):
    """
        train_ind    : indices of the training samples
        perc         : percentage of training set used
        subject_list : list of subject IDs

    return:
        labeled_indices      : indices of the subset of training samples
    """

    train_list = subject_list[train_ind]
    sites = get_subject_score(train_list, score='SITE_ID')
    unique = np.unique(list(sites.values())).tolist()
    site = np.array([unique.index(sites[train_list[x]]) for x in range(len(train_list))])

    labeled_indices = []

    for i in np.unique(site):
        id_in_site = np.argwhere(site == i).flatten()

        num_nodes = len(id_in_site)
        labeled_num = int(round(perc * num_nodes))
        labeled_indices.extend(train_ind[id_in_site[:labeled_num]])

    return labeled_indices


# Load precomputed fMRI connectivity networks
def get_networks(subject_list, kind, atlas_name="aal", variable='connectivity'):
    """
        subject_list : list of subject IDs
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        atlas_name   : name of the parcellation atlas used
        variable     : variable name in the .mat file that has been used to save the precomputed networks


    return:
        matrix      : feature matrix of connectivity networks (num_subjects x network_size)
    """

    all_networks = []
    for subject in subject_list:
        fl = os.path.join(data_folder, subject,
                          subject + "_" + atlas_name + "_" + kind + ".mat")
        matrix = sio.loadmat(fl)[variable]
        all_networks.append(matrix)
    # all_networks=np.array(all_networks)

    idx = np.triu_indices_from(all_networks[0], 1)
    norm_networks = [np.arctanh(mat) for mat in all_networks]
    vec_networks = [mat[idx] for mat in norm_networks]
    matrix = np.vstack(vec_networks)

    return matrix

def get_networks2(subject_list, kind, atlas_name="ho", variable='connectivity'):
    """
        subject_list : list of subject IDs
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        atlas_name   : name of the parcellation atlas used
        variable     : variable name in the .mat file that has been used to save the precomputed networks


    return:
        matrix      : feature matrix of connectivity networks (num_subjects x network_size)
    """

    all_networks = []
    for subject in subject_list:
        fl = os.path.join(data_folder2, subject,
                          subject + "_" + atlas_name + "_" + kind + ".mat")
        matrix = sio.loadmat(fl)[variable]
        all_networks.append(matrix)
    # all_networks=np.array(all_networks)

    idx = np.triu_indices_from(all_networks[0], 1)
    norm_networks = [np.arctanh(mat) for mat in all_networks]
    vec_networks = [mat[idx] for mat in norm_networks]
    matrix = np.vstack(vec_networks)

    return matrix

def get_graph_embedding(subject_list,atlas=None):
    """
        subject_list : list of subject IDs
    return:
        matrix      : feature vec extract by local GNN (num_subjects x embedding_dim)
    """

    all_networks = []
    for subject in subject_list:
        fl = os.path.join('/data0/lsy/SRPBS_new/FBNETGEN_graph_feature/{}/{}.npy'.format(atlas,subject))
        embed = np.load(fl)
        all_networks.append(embed)
    # all_networks=np.array(all_networks)
    matrix = np.vstack(all_networks)

    return matrix

def get_T1features(subject_list):
    """
        subject_list : list of subject IDs
        
    return:
        matrix      : T1 feature matrix of all subjects (num_subjects x network_size)
    """
    from sklearn.preprocessing import MinMaxScaler
    all_networks = []
    for subject in subject_list:
        # fl = os.path.join('/data0/lsy/SRPBS_arrange/T1_sub_feat/', subject +".npy") # old t1 feature dim 256
        # fl = os.path.join('/data0/lsy/SRPBS_new/T1_sub_feat_new/', subject +".npy") # new t1 feature dim 512
        fl = os.path.join('/data0/lsy/SRPBS_new/T1_sub_feat_16/', subject +".npy") # new t1 feature dim 512

        T1_feature = np.load(fl)
        all_networks.append(T1_feature)
    # all_networks=np.array(all_networks)
    # scaler = MinMaxScaler()
    vec_networks = [((vec - np.mean(vec))/np.std(vec)) for vec in all_networks]
    vec_networks = [vec.flatten() for vec in all_networks]
    matrix = np.vstack(vec_networks)

    return matrix

def get_ALFFfeatures(subject_list):
    """
        subject_list : list of subject IDs
        
    return:
        matrix      : T1 feature matrix of all subjects (num_subjects x network_size)
    """

    all_networks = []
    for subject in subject_list:
        fl = os.path.join('/data0/lsy/SRPBS_arrange/T1_sub_feat/', subject +".npy")
        T1_feature = np.load(fl)
        all_networks.append(T1_feature)
    # all_networks=np.array(all_networks)

    vec_networks = [((vec - np.mean(vec))/np.std(vec)) for vec in all_networks]
    vec_networks = [vec.flatten() for vec in all_networks]
    matrix = np.vstack(vec_networks)

    return matrix
# # Construct the adjacency matrix of the population from phenotypic scores
# def create_affinity_graph_from_scores(scores, subject_list):
#     """
#         scores       : list of phenotypic information to be used to construct the affinity graph
#         subject_list : list of subject IDs
#
#     return:
#         graph        : adjacency matrix of the population graph (num_subjects x num_subjects)
#     """
#
#     num_nodes = len(subject_list)
#     graph = np.zeros((num_nodes, num_nodes))
#
#     for l in scores:
#         label_dict = get_subject_score(subject_list, l)
#
#         # quantitative phenotypic scores
#         if l in ['AGE_AT_SCAN']:
#             for k in range(num_nodes):
#                 for j in range(k + 1, num_nodes):
#                     try:
#                         val = abs(float(label_dict[subject_list[k]]) - float(label_dict[subject_list[j]]))
#                         if val < 2:
#                             graph[k, j] += 1
#                             graph[j, k] += 1
#                     except ValueError:  # missing label
#                         pass
#
#         else:
#             for k in range(num_nodes):
#                 for j in range(k + 1, num_nodes):
#                     if label_dict[subject_list[k]] == label_dict[subject_list[j]]:
#                         graph[k, j] += 1
#                         graph[j, k] += 1
#
#     return graph
def create_affinity_graph_from_scores(scores, pd_dict):
    '''
    phenotypic feature # site sex age protocol hand
    '''
    num_nodes = len(pd_dict[scores[0]])
    graph = np.zeros((num_nodes, num_nodes)) # Adjacent matrix

    for l in scores: # l：表型特征指标
        label_dict = pd_dict[l]

        # if l in ['Age','Education (years)']:
        if l in ['Age']:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[k]) - float(label_dict[j])) # 两个节点的相应特征相减
                        if val < 2: # 若年龄差小于2岁，边权值+1，两个都加是因为无向边
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass
        if l in ['FIQ']:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[k]) - float(label_dict[j]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass
        else: # 剩下的都是类别数据
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict[k] == label_dict[j]:
                        graph[k, j] += 1
                        graph[j, k] += 1

    return graph

def get_static_affinity_adj(features, pd_dict):
    '''
    input:
        features: N x img feature dim, extracted image feature
        pd_dict: phenotypic feature dict
    '''
    # pd_affinity = create_affinity_graph_from_scores(['SITE_ID','DSM_IV_TR'], pd_dict) # 使用站点ID以及量表评分进行预剪枝
    pd_affinity = create_affinity_graph_from_scores(['site','age','sex'], pd_dict) # 使用站点ID以及量表评分进行预剪枝

    # site sex age protocol hand
    # pd_affinity = (pd_affinity - pd_affinity.mean(axis=0)) / pd_affinity.std(axis=0)
    # print(pd_affinity)
    # pd_affinity = create_affinity_graph_from_scores(['Sex'], pd_dict)
    distv = distance.pdist(features, metric='correlation') # 计算影像特征的相关性
    dist = distance.squareform(distv) # 一维向量变成邻接矩阵的形式
    sigma = np.mean(dist)
    feature_sim = np.exp(- dist ** 2 / (2 * sigma ** 2)) # 通过高斯分布计算KNN的weight
    # print(feature_sim)
    # print("feature=",feature_sim)
    adj = pd_affinity * feature_sim
    # print("adj=",adj)
    adj = (adj - adj.mean(axis=0)) / adj.std(axis=0)
    # adj = F.normalize(adj)
    # pd_affinity = (pd_affinity - pd_affinity.mean(axis=0)) / pd_affinity.std(axis=0)
    # return pd_affinity # 但是这里返回的只有通过表型数据计算的权重
    return adj # 但是这里返回的只有通过表型数据计算的权重 # 2023-03-07
