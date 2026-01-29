import os
import os.path as path
from collections import defaultdict

import numpy as np
from scipy.ndimage import gaussian_filter, binary_fill_holes, label
from scipy.ndimage import distance_transform_edt
import nibabel as nib
from nilearn.image import resample_to_img
from nibabel.processing import resample_from_to
from nibabel.nifti1 import Nifti1Image
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker
from nilearn.interfaces.fmriprep import load_confounds_strategy
from nilearn.connectome import ConnectivityMeasure
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram, load_tractogram
from dipy.tracking.streamline import transform_streamlines
from regis.core import find_transform, apply_transform
from unravel.utils import get_streamline_density
from unravel.stream import smooth_streamlines
from tqdm import tqdm
import sparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore


def mask_generator(white_matter_probability,
                   gm_path= None, 
                   grey_matter_probability=None, 
                   csf_probability = None, 
                   mask_type = "white", 
                   gm_threshold = 0.3, 
                   smoothing=True):
    """
    Generates a white or grey matter mask in the T1 space of the patient. 
    Functionality starts with just a white matter mask generator

    :param white_matter_probability: str
        Path to a file containing the white matter probability map in T1w space 
        (essential that it is T1 space!! Do not use a mask in MNI space)
    
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
    elif mask_type == "grey": # Eventually this will feature different 
        #behaviour that allows the mask to be computed a little more advanced (as in the commented code above.)
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

    # First crawl through the dMRI folder and get every subject and session
    #  pair for which there is data
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
        bold_path = path.join(bold_fp, 
                              f"sub-{participant}", 
                              session, "func", 
                              f"sub-{participant}_{session}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
        if os.path.exists(bold_path):
            dict_for_results[participant][2]= True

def diffusion_to_t1space(moving_file, static_file, mni= False, smooth = False):
    # TODO this needs to be cleaned up.

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

    # Try to upsample the streamlines

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

def voxel_to_streamline_map_V2(
        streamlines, 
        vol_shape, 
        subsegment:int = 1):
    
    mapping = defaultdict(set)

    failure_count = 0

    points = streamlines.get_data()

    # Creating subpoints
    subpoint = np.linspace(points, np.roll(points, -1, axis=0),
                           subsegment+1, axis=1)
    points = subpoint[:, :-1, :].reshape(points.shape[0]*subsegment, 3)
    del subpoint

    subsegment_offsets = ((streamlines._offsets + streamlines._lengths-1)
                         * subsegment)

    # Try to upsample the streamlines

    min_coord = 100000000
    max_coord = -1000000
    for idx, offset in enumerate(tqdm(subsegment_offsets, "Vox-SL")):
       # Force an integer value for the streamline index
        if idx >= len(subsegment_offsets)-1:
            streamline=points[offset:-subsegment+1]
        else:
            streamline=points[offset:subsegment_offsets[idx+1]-subsegment+1]

        vox = np.round(streamline).astype(np.int32)

        try:
            if vox.min() < min_coord:
                min_coord = vox.min()
            if vox.max() > max_coord:
                max_coord = vox.max()
        except Exception as e:
            print(e)
            print(vox)
            print(offset)
            print(idx)
            print(streamline.shape)
               

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

def mask_to_positions(mask):
    if type(mask) == Nifti1Image:
        wm_data = mask.get_fdata()
    else: 
        wm_data = mask
    
    wm_positions = np.array(np.nonzero(wm_data)).T
    
    return wm_positions

def is_sparse(arr):
    return isinstance(arr, sparse.COO)

def nifti_vs_img(object):
    """
    Checks input to insure it is either a nifti image, 
    or a path to a nifti image
    
    :param object: the object provided
    """
    if type(object) is str:
        img = nib.load(object)
    elif type(object) is Nifti1Image:
        img = object
    else:
        raise TypeError(("Image should be provided as either a path "
            "to an Nifti image, or Nifti image object."))
    return img

def transform_masker(bold_data, masker, bold_filepath, discard_initial):
    if bold_filepath is not None:
        counfounds_df,_= load_confounds_strategy(bold_filepath,
                                            denoise_strategy="simple")
        time_series = masker.fit_transform(bold_data, 
                                           confounds=counfounds_df)    
    else:
        time_series = masker.fit_transform(bold_data)

    return time_series

def create_ROI_time_series(
        atlas, 
        bold_data, 
        discard_initial:int = 3,
        bold_filepath= None, 
        normalise:bool = True):
    
    aal_img = nifti_vs_img(atlas)

    masker = NiftiLabelsMasker(labels_img=aal_img, standardize=normalise)

    return transform_masker(bold_data=bold_data, 
                            masker = masker,
                            discard_initial=discard_initial)

def create_VOX_time_series(
        mask,
        bold_data, 
        discard_initial:int = 3,
        normalise:bool = True):
    
    masker = NiftiMasker(mask_img=mask,
                         standardize=normalise)
    
    return transform_masker(bold_data=bold_data, 
                            masker = masker,
                            discard_initial=discard_initial)
    
def fc_mat_gen(
        timeseries, 
        method: str = "nilearn", 
        kind: str = "correlation"):
        
    # Correlation Matrix
    if method == "nilearn":
        conn_measure = ConnectivityMeasure(kind=kind, standardize=False)
        conn_matrix = conn_measure.fit_transform([timeseries])[0]
    elif method == "custom":
        conn_matrix = matrix_computation(time_series=timeseries)
    else:
        raise ValueError("Enter a valid method: nilearn or custom")
    
    return conn_matrix
    
def connectivity_matrix_generation(bold, 
                                   atlas, 
                                   normalise, 
                                   method= "nilearn", 
                                   kind = "correlation", 
                                   bold_filepath=None, 
                                   return_time_series = False):
    if type(atlas) is str:
        aal_img = nib.load(atlas)
    elif type(atlas) is Nifti1Image:
        aal_img = atlas
    else:
        raise TypeError("The Atlas should be provided as either a path to an image, or the Nifti image object.")
    
    masker = NiftiLabelsMasker(labels_img=aal_img, standardize=normalise)

    if bold_filepath is not None:
        counfounds_df,_= load_confounds_strategy(bold_filepath,
                                            denoise_strategy="simple")
        time_series = masker.fit_transform(bold, 
                                           confounds=counfounds_df)
        
    else:
        time_series = masker.fit_transform(bold)

    
    # Correlation Matrix
    if method == "nilearn":
        conn_measure = ConnectivityMeasure(kind=kind, standardize=False)
        conn_matrix = conn_measure.fit_transform([time_series])[0]
    elif method == "custom":
        conn_matrix = matrix_computation(time_series)
    else:
        raise ValueError("Enter a valid method: nilearn or custom")
    
    if return_time_series:
        return conn_matrix, time_series
    
    return conn_matrix

def matrix_computation(time_series):
    matrix = np.corrcoef(time_series,rowvar=False )
    return matrix

def atlas_registration(atlas_path, 
                       template_file, 
                       reference_file,
                       save_path, 
                       remap = False):
    """
    Registers an atlas to a patient scan. Calculates the transformation based on
    the template file and the reference file (transformation to take the 
    template file to the reference file). Then applies this transformation to 
    the atlas.
    
    :param atlas_path: str
        Filepath to the atlas.
    :param template_file: str
        filepath to the template that the atlas is in. Must be same space as 
        atlas or this will produce garbage.
    :param reference_file: str
        Filepath to the target image. 
    :param save_path: str
        Location to save the registered atlas. First checks this location to see
          if the atlas has already been registered.
    :param remap: str
        Overwrite the loading - if true, the mapping will be calculated again, 
        even if it already exists.
    """
    # First, match the atlas to the patient (this will be a slow step so try 
    # and cache it). Save it somewhere and then just check that filepath.
    img = nifti_vs_img(atlas_path)

    if  remap == False and path.exists(save_path):
        registered_atlas = nib.load(save_path)
    else:
        mapping = find_transform(moving_file= template_file,
                                    static_file= reference_file,
                                    level_iters=[1000, 100, 10],
                                    diffeomorph=False)
        
        registered_atlas = apply_transform(atlas_path, mapping, labels=True)

        # Save the label volume for validation

        out = nib.Nifti1Image(registered_atlas.astype(float), img.affine) 
        out.to_filename(save_path)

def dilate_atlas_labels(atlas, brain_mask, dilation_width):
    """
    Dilates cortical atlas labels to include nearby unlabeled voxels (0),
    constrained by a brain mask.

    Parameters
    ----------
    atlas : np.ndarray (3D, int)
        Cortical atlas with integer labels (0 = unlabeled).
    brain_mask : np.ndarray (3D, bool or int)
        Binary mask of the brain (same shape as atlas).
    dilation_width : float
        Maximum dilation distance in voxels.

    Returns
    -------
    dilated_atlas : np.ndarray (3D, int)
        Atlas with dilated labels.
    """

    # Mask unlabeled voxels inside the brain
    unlabeled = np.where(atlas == 0, 1, 0)
    unlabeled *= brain_mask.astype('int32')

    # Compute distance transform from labeled voxels
    # Also retrieve indices of nearest labeled voxel for each position
    distances, nearest_idx = distance_transform_edt(
        unlabeled, return_indices=True)

    # Copy atlas to output
    dilated_atlas = atlas.copy()

    # For unlabeled voxels within dilation_width, assign nearest label
    within_dilation = unlabeled & (distances <= dilation_width)

    # Map nearest labeled voxel indices back to atlas labels
    coords = np.argwhere(within_dilation)
    for x, y, z in coords:
        nx, ny, nz = nearest_idx[:, x, y, z]
        dilated_atlas[x, y, z] = atlas[nx, ny, nz]

    return dilated_atlas

def visualise_square_mat(matrix, title = "Square Matrix Visualisation"):
        mask = np.triu(np.ones_like(matrix, dtype=bool))

            # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(matrix, mask=mask, cmap=cmap, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.title(title)
        plt.show()

def normalise(array_like, shift_zero: bool = False):
    """
    Docstring for normalise
    
    :param array_like: Matrix or array to normalise
    :param shift_zero: Whether to shift all data such that 0 is the minimum
    :type shift_zero: bool
    """
    scores = zscore(array_like)
    minimum = np.min(scores)
    if minimum < 0:
        scores = scores - minimum
    return scores

def atlas_masker(atlas_data:np.array, target_labels:list):
    """Create a mask of an atlas that retains only the specified label values.
    Parameters
    ----------
    atlas_data : numpy.ndarray
        Array of labeled regions (e.g., an atlas volume or parcellation). Values
        are expected to be label identifiers (commonly integers). The input array's
        shape and dtype are preserved in the returned mask.
    target_labels : Sequence[int]
        Iterable of label values to keep in the output. All entries not matching
        any of these labels will be set to 0 in the returned array.
    Returns
    -------
    numpy.ndarray
        An array with the same shape and dtype as atlas_data where entries that
        match any value in target_labels retain their original label value and
        all other entries are zero.
    Notes
    -----
    - The function prints the indices of nonzero entries for each target label as
      a side effect.
    - If a requested label is not present in atlas_data, it simply has no effect.
    - The input atlas_data is not modified; a new array is returned.
    """
    masked_atlas = np.zeros_like(atlas_data)
    for value in target_labels:
        masked_atlas_v = np.where(atlas_data == value, atlas_data, 0)
        masked_atlas = np.where(masked_atlas_v!=0, masked_atlas_v, masked_atlas)
    
    return masked_atlas

def parse_nib_file(image_obj):
    if type(image_obj) is str:
        pass

def time_slicing(data, slice_length, sliding= False, axis:int = 3):
    """
    Divides a bold signal into time windows. 
    
    :param bold_data: 4D numpy array
        fMRI data with dimensions (x, y, z, time)
    :param slice_length: int
        Length of each time slice

    Notes:
        - Currently discards any excess time points that are not 
            divisible by the slice_length. Consider your slice length 
            value carefully.
    """
    slices = []
    axis_size = data.shape[axis]

    if not sliding:
        num_slices = axis_size // slice_length
        for i in range(num_slices):
            start = i * slice_length
            end = start + slice_length

            slicer = [slice(None)] * data.ndim
            slicer[axis] = slice(start, end)

            slices.append(data[tuple(slicer)])

    else:
        idx = 0
        while idx <= axis_size - slice_length:
            end = idx + slice_length

            slicer = [slice(None)] * data.ndim
            slicer[axis] = slice(idx, end)

            slices.append(data[tuple(slicer)])
            idx += 1

    return np.stack(slices)

def create_masked_T1(t1_file, mask_file, file_path):
    t1_img = nib.load(t1_file)
    t1_data = t1_img.get_fdata()
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata()
    t1_data *= mask_data
    out = nib.Nifti1Image(t1_data, t1_img.affine)
    out.to_filename(file_path)
    return out

