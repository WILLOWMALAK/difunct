# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 11:21:30 2025

@author: nicol
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from regis.core import find_transform, apply_transform
from unravel.analysis import connectivity_matrix
from dipy.io.streamline import load_tractogram

atlas_file = "c:/Users/williamss/Documents/Atlas_Maps/FSL_HCP1065_FA_1mm.nii.gz"
label_file = "c:/Users/williamss/Documents/Atlas_Maps/aal.nii.gz"
subj_file = "c:/Users/williamss/Desktop/TAU_106_ses-0_FA.nii.gz"
trk_file = "c:/Users/williamss/Desktop/TAU_106_ses-0_tractogram.trk"
save_path =  "c:/Users/williamss/Desktop/matrix.npy"
save_fig_path =  "c:/Users/williamss/Desktop/matrix.png"

# Registering the atlas
mapping = find_transform(atlas_file, subj_file, level_iters=[1000, 100, 10],
                         diffeomorph=False)
label_volume = apply_transform(label_file, mapping, labels=True)

img=nib.load(subj_file)
out=nib.Nifti1Image(label_volume.astype(float), img.affine)
out.to_filename('c:/Users/williamss/Desktop/subject_atlas.nii.gz')

# Generating connectivity matrix
trk = load_tractogram(trk_file, 'same')
trk.to_vox()
trk.to_corner()
streamlines = trk.streamlines

matrix = connectivity_matrix(streamlines, label_volume, inclusive=False)

matrix = np.delete(matrix, 0, 0)
matrix = np.delete(matrix, 0, 1)

# Saving output
plt.figure()
plt.imshow(np.log1p(matrix), interpolation='nearest')
plt.show()
np.save(save_path, matrix)
plt.savefig(save_fig_path)
 
# Turn this into a function
def generate_connectivity_matrix(atlas_file, label_file, subj_file, trk_file, save_path=None):

    """
    Creates and stores a connectivity matrix for the subject. 
    
    :param atlas_file: Description
    :param label_file: Description
    :param subj_file: Description
    :param trk_file: Description
    :param save_path: Description
    """

    # Registration
    mapping = find_transform(atlas_file, subj_file, level_iters=[1000, 100, 10],
                         diffeomorph=False)
    label_volume = apply_transform(label_file, mapping, labels=True)

    # Load the subject file image
    img=nib.load(subj_file)
    out=nib.Nifti1Image(label_volume.astype(float), img.affine)
    subject_specific_location = "/subject-num/"
    out.to_filename(save_path + subject_specific_location + "subject_atlas.nii.gz")

    # Make the connectivity matrix
    trk = load_tractogram(trk_file, 'same')
    trk.to_vox()
    trk.to_corner()
    streamlines = trk.streamlines

    matrix = connectivity_matrix(streamlines, label_volume, inclusive=False)

    matrix = np.delete(matrix, 0, 0)
    matrix = np.delete(matrix, 0, 1)

    # Save output
    if save_path !=None:
        np.save(save_path + subject_specific_location + "connectivity_matrix", matrix)
