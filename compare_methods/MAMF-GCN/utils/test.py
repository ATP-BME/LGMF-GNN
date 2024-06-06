import os

import numpy as np
root_folder = '/media/pjc/expriment/mdd_exam/pjc/EV_GCN-master/data/'
data_folder = os.path.join(root_folder, 'ABIDE_pcp')

subject_IDs = np.genfromtxt(os.path.join(root_folder, 'subject_IDs.txt'), dtype=str)
print(subject_IDs)