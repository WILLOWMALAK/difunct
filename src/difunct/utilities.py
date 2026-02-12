import os
import os.path as path
from collections import defaultdict
from itertools import combinations
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt
import nibabel as nib
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
from sparse import DOK
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

def conn_matrices(
        sl_roi_map, 
        vox_sl_map,
        atlas_data,
        mask_positions = None
):
    unique_values = np.unique(atlas_data)
    number_uniques = len(unique_values)
    voxel_cm = {}

    if mask_positions is None:
        for voxel in tqdm(vox_sl_map.keys()):
            voxel_sls = vox_sl_map[voxel]
            temp_array = np.zeros(shape=(number_uniques,number_uniques))
            for sl in voxel_sls:
                start = sl_roi_map[sl][0]
                end = sl_roi_map[sl][1]
                temp_array[start,end] += 1
            temp_array = temp_array + temp_array.T
            conn_mat = sparse.COO.from_numpy(temp_array)
            voxel_cm[voxel] = conn_mat
    else:
        for voxel in tqdm(vox_sl_map.keys()):
            if voxel not in mask_positions:
                continue
            voxel_sls = vox_sl_map[voxel]
            temp_array = np.zeros(shape=(number_uniques,number_uniques))
            for sl in voxel_sls:
                start = sl_roi_map[sl][0]
                end = sl_roi_map[sl][1]
                temp_array[start,end] += 1
                temp_array[end,start] += 1
            conn_mat = sparse.COO.from_numpy(temp_array)
            voxel_cm[voxel] = conn_mat

    return voxel_cm

def conn_matrices_V2(
        sl_roi_map, 
        vox_sl_map,
        atlas_data,
        mask_positions,
        sift_2_weights = None,
        sift_2_mu = None
):
    if not (vox_sl_sanity_check(vox_sl_map)):
        raise ValueError("Voxel mappings are duds.")
    if not (sl_roi_map_sanity_check(sl_roi_map)):
        raise ValueError("SL-ROI map is a dud.")
    
    if sift_2_weights is not None:
        w, mu = load_sift2_weights(
            weights_file=sift_2_weights,
            mu_file=sift_2_mu
        )

    v_coord = []
    rows = []
    cols = []
    values = []
    conn_mat_sz =len(np.unique(atlas_data))-1
    N = len(mask_positions)
    for idx, voxel in enumerate(tqdm(mask_positions, "CM-GEN")):
        voxel_tuple = tuple(voxel)
        if voxel_tuple not in vox_sl_map.keys():
            continue
        for sl in vox_sl_map[voxel_tuple]:
            if sl not in sl_roi_map.keys():   
                continue
            for roi_pair in sl_roi_map[sl]:
                start = roi_pair[0]
                end = roi_pair[1]
                if start == end:
                    continue
                """if (type(start) is not int
                    or type(end) is not int):
                    raise ValueError(f"Non int value: {start}"
                                    f"{end}\n"
                                    f"Type: {type(start)}") """
                voxel_coord = idx
                start_coord = start - 1
                end_coord = end - 1 
                if (start_coord >= conn_mat_sz 
                        or start_coord < 0
                        or end_coord >= conn_mat_sz
                        or end_coord < 0):
                    raise ValueError(f"Start or end coord is out of bounds"
                                    f"Start: {start_coord}\n"
                                    f"End: {end_coord}\n"
                                    f"Bound: {conn_mat_sz}")
                if (voxel_coord < 0 or voxel_coord >= N):
                    raise ValueError(f"Voxel coordinate is out of"
                                    f"bounds: {voxel_coord}")
                v_coord.append(voxel_coord)
                v_coord.append(voxel_coord)
                rows.append(start_coord)
                cols.append(end_coord)
                rows.append(end_coord)
                cols.append(start_coord)
                if sift_2_weights is not None:
                    new_value = w[sl]*mu
                    values.append(new_value)
                    values.append(new_value)
                else:
                    values.append(1) 
                    values.append(1)

    coords = np.vstack([v_coord, rows, cols])
    print(f"Check the coords:\n"
          f"Shape {coords.shape}\n"
          f"Max: {coords.max()}\n"
          f"Min: {coords.min()}")

    all_cms = sparse.COO(
        coords=coords,
        data = values,
        shape=(N,conn_mat_sz, conn_mat_sz),
        has_duplicates=True
    )

    return all_cms
   
def complete_data_compiler(dfmri_fp, 
                           bold_fp,
                             data_filepath=None):
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

def create_masked_T1(t1_file, mask_file, file_path):
    t1_img = nib.load(t1_file)
    t1_data = t1_img.get_fdata()
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata()
    t1_data *= mask_data
    out = nib.Nifti1Image(t1_data, t1_img.affine)
    out.to_filename(file_path)
    return out

def create_ROI_time_series(
        atlas, 
        bold_data, 
        discard_initial:int = 3,
        bold_filepath= None, 
        normalise:bool = True):
    
    aal_img = nifti_vs_img(atlas)

    masker = NiftiLabelsMasker(labels_img=aal_img, standardize=normalise)

    time_series = transform_masker(bold_data=bold_data, 
                            masker = masker,
                            bold_filepath=bold_filepath,
                            discard_initial=discard_initial)
    
    return time_series

def create_VOX_time_series(
        mask,
        bold_data, 
        bold_filepath = None,
        discard_initial:int = 3,
        normalise:bool = True):
    
    masker = NiftiMasker(mask_img=mask,
                         standardize=normalise)
    
    time_series = transform_masker(bold_data=bold_data, 
                            masker = masker,
                            bold_filepath=bold_filepath,
                            discard_initial=discard_initial)
    
    return time_series

def streamline_registration(
        moving_file, 
        static_file, 
        trk_file,
        mni= False, 
        smooth = False, 
        save = False):
    """"""
    # TODO this needs to be cleaned up.
    # Moving file is in diffusion space
    if mni:
    # Ignore
        static_file = 'C:/Users/nicol/Documents/Doctorat/Data/Atlas_Maps/FSL_HCP1065_FA_1mm.nii.gz'
        mapping = find_transform(static_file, moving_file, diffeomorph=False)
    else:
    # T1 file
        mapping = find_transform(static_file, moving_file, only_affine=True)

    # For every tract you want to register
    for r in ["1"]:
    # Or hardcode the filename
        trk = load_tractogram(trk_file, 'same')

        stream_reg = transform_streamlines(trk.streamlines,
                                       # np.linalg.inv(mapping.affine))
                                       mapping.affine)

        sft_reg = StatefulTractogram(
            stream_reg, 
            nib.load(static_file), 
            Space.RASMM)

    # trk_new = StatefulTractogram(streams, trk, Space.VOX,
    #                                  origin=Origin.TRACKVIS)

        if mni:
            out_file = trk_file[:-4]+'_mni.trk'
        else:
            out_file = trk_file[:-4]+'_T1.trk'
        if save is True:
            if type(save) is str:
                save_tractogram(
                sft = trk,
                filename=save
            )
            else:
                save_tractogram(
                    sft = trk,
                    filename=out_file
                )
        if smooth:
        # For visualization, not computing
            smooth_streamlines(out_file, out_file=out_file[:-4]+'_smoothed.trk',
                           iterations=50)
        
        return sft_reg

def mask_generator(white_matter_probability,
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

        gm_img = nib.load(grey_matter_probability)
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

def mask_to_positions(mask):
    if type(mask) == Nifti1Image:
        wm_data = mask.get_fdata()
    else: 
        wm_data = mask
    
    wm_positions = np.array(np.nonzero(wm_data)).T
    
    return wm_positions

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

def sl_to_roi_map(
        streamlines,
        atlas, 
        only_endpoints = True
):
    vol_shape = atlas.shape
    mapping = defaultdict(lambda: defaultdict(list))
    points = streamlines.get_data()
    start_points = streamlines._offsets
    if only_endpoints:
        for idx in range(len(start_points)):
            start = start_points[idx]
            end   = start_points[idx+1] if idx + 1 < len(start_points) else len(points)
            sl_start = points[start].astype(np.int32)
            sl_end   = points[end - 1].astype(np.int32)
            vxl_coords_start = tuple(sl_start)
            vxl_coords_end = tuple(sl_end)
            if (not value_in_bounds(vxl_coords_start, vol_shape) or
                not value_in_bounds(vxl_coords_end, vol_shape)):
                continue
            start_roi = int(atlas[vxl_coords_start])
            end_roi = int(atlas[vxl_coords_end])
            if (start_roi == 0 or end_roi == 0):
                continue
            #mapping[start_roi][end_roi].append(idx)
            mapping[idx] = [tuple([start_roi, end_roi])]
    else:
        for idx in range(len(start_points)):
            start = start_points[idx]
            end   = start_points[idx+1] if idx + 1 < len(start_points) else len(points)
            paired_rois = []
            sl = points[start:end].astype(np.int32)
            x, y, z = sl.T
            crossed_labels = np.unique(atlas[x,y,z]).astype(np.int32)
            if crossed_labels.min() < 0:
                raise ValueError(f"Less than 0 value has appeared")
            for comb in combinations(crossed_labels, r=2):
                if 0 in comb:
                    continue
                paired_rois.append(comb)
            mapping[idx] = paired_rois
    return mapping

def sl_roi_map_sanity_check(sl_roi_map):
    sum = 0
    for key in sl_roi_map.keys():
        if type(sl_roi_map[key]) != list:
            sum += 1
    if sum != 0:
        return False
    else:
        return True

def value_in_bounds(
        coords, 
        dimensions
):
    if (
    0 <= coords[0] < dimensions[0] and
    0 <= coords[1] < dimensions[1] and
    0 <= coords[2] < dimensions[2]   
    ):
        return  True
    else:
        False

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

def vox_sl_sanity_check(vox_sl):
    sum = 0
    for key in vox_sl.keys():
        if len(vox_sl[key]) != 0:
            sum += 1
    if sum != 0:
        return True
    else:
        return False

def is_sparse(arr):
    return isinstance(arr, sparse.COO)

def transform_masker(bold_data, 
                     masker,
                     discard_initial, 
                     bold_filepath = None):
    if bold_filepath is not None:
        counfounds_df,_= load_confounds_strategy(bold_filepath,
                                            denoise_strategy="simple")
        time_series = masker.fit_transform(bold_data, 
                                           confounds=counfounds_df)    
    else:
        time_series = masker.fit_transform(bold_data)

    return time_series[discard_initial:]

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
                       save_path = None):
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
    """
    # First, match the atlas to the patient (this will be a slow step so try 
    # and cache it). Save it somewhere and then just check that filepath.
    atl_img = nifti_vs_img(atlas_path)
    template_img = nifti_vs_img(template_file)
    if not np.allclose(atl_img.affine, template_img.affine,rtol=1e-3):
        raise ValueError(f"The template file and the atlas are not aligned")

    if  save_path is not None and path.exists(save_path):
        out = nib.load(save_path)
        return out
    else:
        mapping = find_transform(
            moving_file= template_file,
            static_file= reference_file,
            level_iters=[1000, 100, 10],
            diffeomorph=False
            )
        
        registered_atlas = apply_transform(
            atlas_path,
            mapping, 
            labels=True)

        # Save the label volume for validation
        reference_img = nifti_vs_img(reference_file)
        out = nib.Nifti1Image(
            registered_atlas.astype(float), 
            reference_img.affine) 

        if save_path is None:
            return out
        else:
            out.to_filename(save_path)
            return out

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

def split_nifti_to_visualise(original_fp:str):
    original = nib.load(original_fp)
    original_data = original.get_fdata()
    pos_data = np.where(original_data>0, original_data, 0)
    neg_data = np.where(original_data<0, np.abs(original_data), 0)
    neg_fp = original_fp[:-7] + "_neg.nii.gz"
    pos_fp = original_fp[:-7] + "_pos.nii.gz"
    pos_img = nib.Nifti1Image(pos_data, original.affine)
    pos_img.to_filename(pos_fp)
    neg_img = nib.Nifti1Image(neg_data, original.affine)
    neg_img.to_filename(neg_fp)

def trk_vs_filepath(trk_obj):
    if type(trk_obj) is StatefulTractogram:
        return trk_obj
    elif type(trk_obj) is str:
        return load_tractogram(trk_obj, "same")
    else:
        raise ValueError(f"Expected either a stateful tractogram," 
                         f"or a path to a trk file")

def trk2tck(input_file: str, bounding = True):
    tract = load_tractogram(input_file, 'same', bbox_valid_check=bounding)
    save_tractogram(tract, input_file[:-3]+'tck', bbox_valid_check=bounding)

def simple_weighting(SC, FC):
   """Simple element wise multiplcation of structural and functional connectivity"""
   denom = np.max(SC)
   if denom == 0:
       raise ValueError("Denominator is 0. Structural Connectivity matrix is all zero")
   normed_SC = SC/denom

   # Simple element wise weighting
   resultant = np.multiply(normed_SC, FC)

   return resultant

def load_sift2_weights(weights_file: str, mu_file: str):

    text = []

    f = open(weights_file, "r")
    for x in f:
        text.append(x)
    f.close()
    
    f = open(mu_file, "r")
    for x in f:
        mu=float(x)
    f.close()

    w = np.array([float(i) for i in text[1].split(' ')], dtype='float32')
    
    return w,mu

def sparse_equality(sparse_1, sparse_2):
    """
   Simple function to check for equality between 
   two sparse arrays
    
    :param sparse_1: sparse.COO array
        First array to check
    :param sparse_2: sparse.COO array
        Second array to check
    """
    if sparse_1.shape != sparse_2.shape:
        return False
    if sparse_1.nnz != sparse_2.nnz:
        return False
    if (sparse_1-sparse_2).nnz !=0:
        return False
    else:
        return True
