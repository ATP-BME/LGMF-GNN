# Copyright (c) 2019 Mwiza Kunda
# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implcd ied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import pickle
import os
import warnings
import glob
import csv
import re
import numpy as np
import scipy.io as sio
import sys
from nilearn import connectome
import pandas as pd
from scipy.spatial import distance
from scipy import signal
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

# Input data variables

# root_folder = '/home/tux/wjy/final_code/BrainGNN/BrainGNN_Pytorch/data/'
# data_folder = os.path.join(root_folder, 'ABIDE_pcp/cpac/tmp')
# phenotype = os.path.join(root_folder, 'ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')
root_folder = '/home/sjtu/liushuyu/project/BrainGNN_SRPBS/BrainGNN_Pytorch/data'
ALFF_data_folder = '/data0/lsy/SRPBS_arrange/ALFF_ROI/'
data_folder = '/data0/lsy/SRPBS_arrange/ROI_signal/'
# phenotype = os.path.join(root_folder, 'SRPBS/participants.csv')
phenotype = "/data0/lsy/SRPBS_arrange/ROI_signal/participants.csv"
atlas_name='cc200'


def fetch_filenames(subject_IDs, file_type, atlas):
    """
        subject_list : list of short subject IDs in string format
        file_type    : must be one of the available file types
        filemapping  : resulting file name format
    returns:
        filenames    : list of filetypes (same length as subject_list)
    """

# datapath "/data0/lsy/SRPBS_arrange/protocol_5/Results/ROISignals_FunImgARCSF/ROISignals_sub_0526.mat"
# funimg "/data0/lsy/SRPBS_arrange/protocol_3/FunImgARCSFsymS/"
    filemapping = {'func_preproc': '_func_preproc.nii.gz',
                   'rois_' + atlas: '.mat'}
    # The list to be filled
    filenames = []

    # Fill list with requested file paths
    for i in range(len(subject_IDs)):
        os.chdir(data_folder)
        try:
            try:
                os.chdir(data_folder)
                filenames.append(glob.glob('*' + subject_IDs[i] + filemapping[file_type])[0])
            except:
                os.chdir(data_folder + '/' + subject_IDs[i])
                filenames.append(glob.glob('ROISignals_'+ subject_IDs[i] + filemapping[file_type])[0])
        except IndexError:
            filenames.append('N/A')
    return filenames


# Get timeseries arrays for list of subjects
def get_timeseries(subject_list, atlas_name, silence=False):
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
        if silence != True:
            print("Reading timeseries file %s" % fl)
        timeseries.append(np.loadtxt(fl, skiprows=0))

    return timeseries

def get_timeseries_mat(subject_list,mat_root,silence=False):
    """
        subject_list : list of short subject IDs in string format
        mat_root: root folder of roi signal mat files
    returns:
        time_series  : list of timeseries arrays, each of shape (timepoints x regions)
    """

# "/data0/lsy/SRPBS_arrange/ROI_signal/sub_0236/ROISignals_sub_0236.mat"
    timeseries = []
    for i in range(len(subject_list)):
        # subject_folder = os.path.join(data_folder, subject_list[i])
        # ro_file = [f for f in os.listdir(subject_folder) if f.endswith('_rois_' + atlas_name + '.1D')]
        ro_file = 'ROISignals_'+subject_list[i]+'.mat'
        fl = os.path.join(mat_root,subject_list[i], ro_file)
        matrix = sio.loadmat(fl)['ROISignals']
        if silence != True:
            print("Reading timeseries file %s" % fl,'\t',matrix.shape)
        timeseries.append(matrix)

    return timeseries

def get_ALFF_mat(subject_list,mat_root,silence=False):
    """
        subject_list : list of short subject IDs in string format
        mat_root: root folder of roi ALFF signal mat files
    returns:
        time_series  : list of timeseries arrays, each of shape (timepoints x regions)
    """

# "/data0/lsy/SRPBS_arrange/ALFF_band1_ROI/ROISignals_ROISignal_sub_0301.mat"
    timeseries = []
    for i in range(len(subject_list)):
        subject_folder = os.path.join(data_folder, subject_list[i])
        ro_file = 'ROISignals_'+subject_list[i]+'.mat'
        # print(ro_file)
        fmir_roi_path = os.path.join(subject_folder, ro_file)
        fmri_sig= sio.loadmat(fmir_roi_path)['ROISignals']
        ro_file = 'ROISignals_ROISignal_'+subject_list[i]+'.mat'
        fl = os.path.join(mat_root, ro_file)
        fl_band1 = os.path.join(mat_root[0:-1]+'_band1', ro_file)
        matrix = sio.loadmat(fl)['ROISignals']
        matrix_band1 = sio.loadmat(fl_band1)['ROISignals']
        matrix = np.concatenate((fmri_sig,matrix,matrix_band1))
        if silence != True:
            print("Reading timeseries file %s" % fl,'\t',matrix.shape)
        timeseries.append(matrix)

    return timeseries

#  compute connectivity matrices
def subject_connectivity(timeseries, subjects, atlas_name, kind, iter_no='', seed=1234,
                         n_subjects='', save=True, save_path=ALFF_data_folder):
    """
        timeseries   : timeseries table for subject (timepoints x regions)
        subjects     : subject IDs
        atlas_name   : name of the parcellation atlas used
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        iter_no      : tangent connectivity iteration number for cross validation evaluation
        save         : save the connectivity matrix to a file
        save_path    : specify path to save the matrix if different from subject folder
    returns:
        connectivity : connectivity matrix (regions x regions)
    """

    if kind in ['TPE', 'TE', 'correlation','partial correlation']:
        if kind not in ['TPE', 'TE']:
            conn_measure = connectome.ConnectivityMeasure(kind=kind)
            connectivity = conn_measure.fit_transform(timeseries)
        else:
            if kind == 'TPE':
                conn_measure = connectome.ConnectivityMeasure(kind='correlation')
                conn_mat = conn_measure.fit_transform(timeseries)
                conn_measure = connectome.ConnectivityMeasure(kind='tangent')
                connectivity_fit = conn_measure.fit(conn_mat)
                connectivity = connectivity_fit.transform(conn_mat)
            else:
                conn_measure = connectome.ConnectivityMeasure(kind='tangent')
                connectivity_fit = conn_measure.fit(timeseries)
                connectivity = connectivity_fit.transform(timeseries)

    if save:
        if kind not in ['TPE', 'TE']:
            for i, subj_id in enumerate(subjects):
                subject_file = os.path.join(save_path, subj_id,
                                            subj_id + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat')
                if not os.path.exists(os.path.join(save_path, subj_id)):
                    os.mkdir(os.path.join(save_path, subj_id))
                sio.savemat(subject_file, {'connectivity': connectivity[i]})
            return connectivity
        else:
            for i, subj_id in enumerate(subjects):
                subject_file = os.path.join(save_path, subj_id,
                                            subj_id + '_' + atlas_name + '_' + kind.replace(' ', '_') + '_' + str(
                                                iter_no) + '_' + str(seed) + '_' + validation_ext + str(
                                                n_subjects) + '.mat')
                sio.savemat(subject_file, {'connectivity': connectivity[i]})
            return connectivity_fit


# Get the list of subject IDs

def get_ids(txt_name='subject_IDs.txt',num_subjects=None):
    """
    return:
        subject_IDs    : list of all subject IDs
    """

    # subject_IDs = np.genfromtxt(os.path.join(data_folder,txt_name), dtype=str)
    subject_IDs = np.genfromtxt(txt_name, dtype=str)

    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]

    return subject_IDs


# Get phenotype values for a list of subjects
def get_subject_score(subject_list, score):
    scores_dict = {}

    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['participant_id'].replace('-','_') in subject_list:
                if score == 'diag':
                    print(row[score])
                    if row[score] == '0':
                        scores_dict[row['participant_id'].replace('-','_')] = 0
                    if row[score] == '2':
                        scores_dict[row['participant_id'].replace('-','_')] = 1
                    # if row[score] == '99':
                    #     scores_dict[row['participant_id'].replace('-','_')] = 7
                    # if row[score] == '4':
                    #     scores_dict[row['participant_id'].replace('-','_')] = 3
                    # if row[score] == '5':
                    #     scores_dict[row['participant_id'].replace('-','_')] = 4
                    # if row[score] == '7':
                    #     scores_dict[row['participant_id'].replace('-','_')] = 5
                    # if row[score] == '8':
                    #     scores_dict[row['participant_id'].replace('-','_')] = 6
                    # if row[score] == '0' or row[score] == '1' or row[score] == '2':
                    #     scores_dict[row['participant_id'].replace('-','_')] = int(row[score])
                else:
                    print('wrong diagnose column name')

    return scores_dict


# preprocess phenotypes. Categorical -> ordinal representation
def preprocess_phenotypes(pheno_ft, params):
    if params['model'] == 'MIDA':
        ct = ColumnTransformer([("ordinal", OrdinalEncoder(), [0, 1, 2])], remainder='passthrough')
    else:
        ct = ColumnTransformer([("ordinal", OrdinalEncoder(), [0, 1, 2, 3])], remainder='passthrough')

    pheno_ft = ct.fit_transform(pheno_ft)
    pheno_ft = pheno_ft.astype('float32')

    return (pheno_ft)


# create phenotype feature vector to concatenate with fmri feature vectors
def phenotype_ft_vector(pheno_ft, num_subjects, params):
    gender = pheno_ft[:, 0]
    if params['model'] == 'MIDA':
        eye = pheno_ft[:, 0]
        hand = pheno_ft[:, 2]
        age = pheno_ft[:, 3]
        fiq = pheno_ft[:, 4]
    else:
        eye = pheno_ft[:, 2]
        hand = pheno_ft[:, 3]
        age = pheno_ft[:, 4]
        fiq = pheno_ft[:, 5]

    phenotype_ft = np.zeros((num_subjects, 4))
    phenotype_ft_eye = np.zeros((num_subjects, 2))
    phenotype_ft_hand = np.zeros((num_subjects, 3))

    for i in range(num_subjects):
        phenotype_ft[i, int(gender[i])] = 1
        phenotype_ft[i, -2] = age[i]
        phenotype_ft[i, -1] = fiq[i]
        phenotype_ft_eye[i, int(eye[i])] = 1
        phenotype_ft_hand[i, int(hand[i])] = 1

    if params['model'] == 'MIDA':
        phenotype_ft = np.concatenate([phenotype_ft, phenotype_ft_hand], axis=1)
    else:
        phenotype_ft = np.concatenate([phenotype_ft, phenotype_ft_hand, phenotype_ft_eye], axis=1)

    return phenotype_ft


# Load precomputed fMRI connectivity networks
def get_networks(subject_list, kind, iter_no='', seed=1234, n_subjects='', atlas_name="aal",
                 variable='connectivity'):
    """
        subject_list : list of subject IDs
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation,T1_feat,correlation+T1
        atlas_name   : name of the parcellation atlas used
        variable     : variable name in the .mat file that has been used to save the precomputed networks
    return:
        matrix      : feature matrix of connectivity networks (num_subjects x network_size)
    """
    print('node feature kind:',kind)
    all_networks = []
    for subject in subject_list:
        if len(kind.split()) == 2:
            kind = '_'.join(kind.split())
        if kind == 'correlation':
            fl = os.path.join(ALFF_data_folder, subject,
                                subject + "_" + atlas_name + "_" + kind.replace(' ', '_') + ".mat")
        if kind == 'partial_correlation':
            fl = os.path.join(data_folder, subject,
                                subject + "_" + atlas_name + "_" + kind.replace(' ', '_') + ".mat")
        if kind == 'T1_feat':
            fl = "/data0/lsy/SRPBS_arrange/T1_sub_feat/{}.npy".format(subject)
        if kind == 'correlation+T1':
            fl = [os.path.join(ALFF_data_folder, subject,subject + "_" + atlas_name + "_" + 'correlation' + ".mat"),\
                "/data0/lsy/SRPBS_arrange/T1_sub_feat/{}.npy".format(subject)]
    # for subject in subject_list:
    #     if len(kind.split()) == 2:
    #         kind = '_'.join(kind.split())
    #     #在这里读取mat数据,到时候在这里改就行了

    #     fl = os.path.join(data_folder, subject,
    #                           "ROICorrelation" + "_" + subject + ".mat")

        print('fl is ', fl)

        # if kind != 'T1_feat':
        if kind in ['TE','TPE','correlation','partial_correlation']:
            matrix = sio.loadmat(fl)[variable]
        elif kind == 'T1_feat':
            matrix = np.load(fl)
        else:
            matrix_T1 = np.load(fl[1])
            matrix_corr = sio.loadmat(fl[0])[variable]
            matrix_corr = np.arctanh(matrix_corr)
            matrix = np.concatenate([matrix_T1,matrix_corr],axis=1)
        all_networks.append(matrix)

    if kind in ['TE', 'TPE','T1_feat','correlation+T1']:
        norm_networks = [mat for mat in all_networks]
    else:
        norm_networks = [np.arctanh(mat) for mat in all_networks]
    #networks没问题
    networks = np.stack(norm_networks)

    with open('networks.pickle', 'wb') as handle:
        pickle.dump(networks, handle, protocol=2)

    return networks

