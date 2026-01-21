import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, binary_fill_holes, label
from nilearn.image import resample_to_img
from nibabel.processing import resample_from_to
import os
import os.path as path
from unravel.stream import smooth_streamlines
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram, load_tractogram
from dipy.tracking.streamline import transform_streamlines
from regis.core import find_transform
from unravel.utils import get_streamline_density
from collections import defaultdict
from tqdm import tqdm
from nibabel.nifti1 import Nifti1Image

gm_path = "/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_label-GM_probseg.nii.gz"
wm_path = "/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_label-WM_probseg.nii.gz"
csf_path = "/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_label-CSF_probseg.nii.gz"
save_name = "/Users/sam/Desktop/sub-TAU001/sub-TAU001_space-T1w_label-GM_mask.nii.gz"
wm_save_name = "/Users/sam/Desktop/sub-TAU001/sub-TAU001_space-T1w_label-WM_mask.nii.gz"
bold_path = "/Users/sam/Desktop/sub-TAU001/ses-2/func/sub-TAU001_ses-2_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"


def mask_generator(white_matter_probability, grey_matter_probability=None, csf_probability = None, mask_type = "white", gm_threshold = 0.3, smoothing=True):
    """
    Generates a white or grey matter mask in the T1 space of the patient. Functionality starts with just a white matter mask generator

    :param white_matter_probability: str
        Path to a file containing the white matter probability map in T1w space (essential that it is T1 space!! Do not use a mask in MNI space)
    
    """
    white_matter_img = nib.load(white_matter_probability)
    wm_data = white_matter_img.get_fdata()

    if mask_type == "white":

        # Optional Smoothing.
        if smoothing == True:
            wm_smooth = gaussian_filter(wm_data, sigma = 1.0)
        else:   
            wm_smooth = wm_data

        wm_mask = (wm_smooth > 0.1) # This is commonly used apparently.
        out = nib.Nifti1Image(wm_mask, white_matter_img.affine, white_matter_img.header) 
        return out
    elif mask_type == "grey": # Eventually this will feature different behaviour that allows the mask to be computed a little more advanced (as in the commented code above.)
        if grey_matter_probability == None or csf_probability == None:
            raise ValueError("Both grey matter probability and csf probability must be provided.")

        gm_img = nib.load(gm_path)
        gm_data = gm_img.get_fdata()  
        csf_img = nib.load(csf_probability)
        csf_data = csf_img.get_fdata()
        
        gm_smooth = gaussian_filter(gm_data, sigma=1.0)
        gm_mask = (
                    (gm_smooth > gm_threshold) &
                    (gm_smooth > wm_data) & 
                    (gm_smooth > csf_data)         
                        ).astype("uint8")
        out =  nib.Nifti1Image(gm_mask, gm_img.affine, gm_img.header) 
        return out
    else: 
        print(f"Invalid mask type specified: {mask_type}. Valid values are \"white\" and \"grey\"")


def complete_data_compiler(dfmri_fp, bold_fp, data_filepath=None):
    """
    Docstring for complete_data_compiler
    
    :param dfmri_fp: Description
    :param bold_fp: str
        Filepath to the derivative folder where preprocessed fMRI data lives
    :param data_filepath: Description
    """

    # First crawl through the dMRI folder and get every subject and session pair for which there is data
    dict_for_results = {}
    for folder in os.listdir(dfmri_fp):
        split_name = folder.split(sep = "_")
        subj_number = split_name[1]
        session = split_name[-1]
        identifier = f"TAU{subj_number.zfill(3)}"
        available_data = [session, True, False, False]

        dict_for_results[identifier] = available_data
    
       
    # Iterate through all the patients that we have dmri data for

    for participant in dict_for_results:
        session = dict_for_results[participant][0]
        bold_path = path.join(bold_fp, f"sub-{participant}", session, "func", f"sub-{participant}_{session}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
        if os.path.exists(bold_path):
            dict_for_results[participant][2]= True


    

def diffusion_to_t1space(moving_file, static_file, mni= False, smooth = False):

    # Diffusion space file
    moving_file = '/Users/sam/Desktop/sub-TAU001/TAU_1_ses-2_FA.nii.gz'

    if mni:
    # Ignore
        static_file = 'C:/Users/nicol/Documents/Doctorat/Data/Atlas_Maps/FSL_HCP1065_FA_1mm.nii.gz'
        mapping = find_transform(static_file, moving_file, diffeomorph=False)
    else:
    # T1 file
        static_file = '/Users/sam/Desktop/sub-TAU001/anat/sub-TAU001_desc-preproc_T1w.nii.gz'
        mapping = find_transform(static_file, moving_file, only_affine=True)


    # For every tract you want to register
    for r in ["1"]:
    # Or hardcode the filename
        trk_file = '/Users/sam/Desktop/TAU_1_ses-2_tractogram.trk'
        #trk_file = 'C:/Users/nicol/Desktop/temp_anais/10_'+r+'.trk'

        trk = load_tractogram(trk_file, 'same')

        stream_reg = transform_streamlines(trk.streamlines,
                                       # np.linalg.inv(mapping.affine))
                                       mapping.affine)

        sft_reg = StatefulTractogram(
        stream_reg, nib.load(static_file), Space.RASMM)

    # trk_new = StatefulTractogram(streams, trk, Space.VOX,
    #                                  origin=Origin.TRACKVIS)

        if mni:
            out_file = trk_file[:-4]+'_mni.trk'
        else:
            out_file = trk_file[:-4]+'_T1.trk'

        save_tractogram(sft_reg, out_file, bbox_valid_check=False)

        if smooth:
        # For visualization, not computing
            smooth_streamlines(out_file, out_file=out_file[:-4]+'_smoothed.trk',
                           iterations=50)
            
def voxel_to_streamline_map(streamlines, vol_shape):
    mapping = defaultdict(set)

    failure_count = 0

    min_coord = 100000000
    max_coord = -1000000
    for idx, streamline in enumerate(tqdm(streamlines, "Vox-SL")):
        # Force an integer value for the streamline index
        vox = np.round(streamline).astype(np.int32)

        if vox.min() < min_coord:
            min_coord = vox.min()
        if vox.max() > max_coord:
            max_coord = vox.max()
               

        # Remove points outside the shape
        valid_vox = (
                        (vox[:,0] >= 0) & (vox[:, 0] < vol_shape[0]) &
                        (vox[:,1] >= 0) & (vox[:, 1] < vol_shape[1]) &
                        (vox[:,2] >= 0) & (vox[:, 2] < vol_shape[2])
        )
        if np.sum(valid_vox) < 3:
            failure_count += 1
        vox = vox[valid_vox]

        # One streamline should only be counted once per voxel
        for v in map(tuple, np.unique(vox, axis=0)):
            mapping[v].add(idx)
            
    # Convert sets → lists for downstream use
    return {k: list(v) for k, v in mapping.items()}

def generate_masks(wm_mask, test_masks = False):
    if test_masks:
        mask_1 = nib.load("/Users/sam/Desktop/sub-TAU001/one_white_matter_mask.nii.gz")
        mask_2 = nib.load("/Users/sam/Desktop/sub-TAU001/two_white_matter_mask.nii.gz")
        return np.stack([mask_1.get_fdata(), mask_2.get_fdata()], axis=0)
    else:
        if type(wm_mask) == Nifti1Image:
            wm_data = wm_mask.get_fdata()
        else: 
            wm_data = wm_mask
       
        wm_positions = np.array(np.nonzero(wm_data)).T
        #n = len(wm_positions)

        """ # Make an array to store the masks. It should be n long, and then the same shape as the original white matter mask
        output = np.zeros((n,) + wm_data.shape, dtype=wm_data.dtype)

        # For each index in the wm_positions array, create a single 1.0 value in the mask.
        for i, idx in tqdm(enumerate(wm_positions), "\tGenerating the white matter masks"):
            output[i][tuple(idx)] = 1.0 """
        
        return wm_positions
