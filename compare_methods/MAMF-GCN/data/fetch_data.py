from nilearn import datasets
import ABIDEParser as Reader
import os
import shutil
from dataloader import dataloader

# Selected pipeline
pipeline = 'cpac'

dl = dataloader()
# raw_features, y, nonimg = dl.load_data()

# Input data variables
num_subjects = 532
# num_subjects = raw_features.shape[0]  # Number of subjects
#root_folder = '/bigdata/fMRI/ABIDE/'
root_folder = '/media/pjc/expriment/mdd_exam/pjc/EV_GCN-MDD/data'
data_folder = os.path.join(root_folder, 'MDD_pcp/mdd_data_ho2')

# Files to fetch
files = ['rois_ho']

filemapping = {'rois_ho': '.mat'}

# if not os.path.exists(data_folder): os.makedirs(data_folder)
# shutil.copyfile('./MDDsubject_IDs.txt', os.path.join(data_folder, 'MDDsubject_IDs.txt'))

# Download database files
# abide = datasets.fetch_abide_pcp(data_dir=root_folder, n_subjects=num_subjects, pipeline=pipeline,
#                                  band_pass_filtering=True, global_signal_regression=False, derivatives=files)


subject_IDs = Reader.get_ids(num_subjects)
subject_IDs = subject_IDs.tolist()
# print(len(subject_IDs))

# Create a folder for each subject
for s, fname in zip(subject_IDs, Reader.fetch_filenames(subject_IDs, files[0])):
    print("fname=",fname)
    subject_folder = os.path.join(data_folder, s)
    print("subject_folder=",subject_folder)
    if not os.path.exists(subject_folder):
        os.mkdir(subject_folder)

    # Get the base filename for each subject
    base = fname.split('.')[0]
    print("base=",base)

    # Move each subject file to the subject folder
    for fl in files:
        if not os.path.exists(os.path.join(subject_folder,base + filemapping[fl])):
            shutil.move(base + filemapping[fl], subject_folder)

time_series = Reader.get_timeseries(subject_IDs, 'ho')

# Compute and save connectivity matrices
for i in range(len(subject_IDs)):
        Reader.subject_connectivity(time_series[i], subject_IDs[i], 'ho', 'correlation')

