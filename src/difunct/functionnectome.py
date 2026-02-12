import os
from os import path
from collections import defaultdict
import json
import numpy as np
from numpy import tensordot
import sparse
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import matplotlib.pyplot as plt
from dipy.tracking.utils import target, density_map
from dipy.tracking.streamline import select_by_rois, transform_streamlines
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
from unravel.utils import get_streamline_density
from nilearn.maskers import (NiftiLabelsMasker, 
                             NiftiMasker)
from nilearn import image, masking
from nilearn.regions import signals_to_img_labels
from utilities import (voxel_to_streamline_map, 
                       mask_generator, 
                       is_sparse, 
                       mask_to_positions,
                       nifti_vs_img)

NOISE_OFFSET = 5

def target_1(trk, mask, affine):  
    """
    Generates the subset of all streamlines that pass 
    through a given voxel/masked value.
    
    :param trk: The trk file containing the tracts
    :param mask: A brain mask highlighting a region that we want to
        extract streamlines for
    :param affine: For now, unused. May be needed if the function is 
        modified to take a tck and then the affine needs to be passed.

    :return rel_streamlines: A tractogram with the streamlines that 
        pass through the masked location.
    """
    streamlines = trk.streamlines
    rel_streamlines = target(streamlines, affine, mask)
    return rel_streamlines

def probability_maps(trk, mask_array, mask_debug=False):
    """
    Compute the probability that a given voxel is connected to any 
    other voxel. This method iterates through masks.

    This method will take an absurd amount of time to run. In
    the order of roughly 300 hours. May need to do groups of 2x2x2 voxels? 
    This would bring down by a factor of 8.
    
    :param wm_mask: Description
    :param trk: Description
    """
    # Initial attempt: Iterate through all and attempt the procedure. 
    # If they all come back empty, something is wrong.
    successes = 0
    failures = 0
    streamline_count = []
    for idx, mask in tqdm(enumerate(mask_array[0:999])):
        try:
            voxel_results = target_1(trk, mask, np.eye(4))
            trk_new = trk.from_sft(voxel_results, trk)
            streamline_count.append(len(trk_new.streamlines))
            density = get_streamline_density(trk_new, resolution_increase=1)
            successes += 1
        except (ValueError, IndexError) as e:
            failures += 1

    print(
        f"The number of successes: {successes}\n"
        f"Streamline Characteristics\n"
        f" Mean: {np.mean(streamline_count)}\n"
        f" Max: {np.max(streamline_count)}\n"
        f"Streamline counts:\n"
        f"{streamline_count}")

def vectorised_probability_maps(
        registered_atlas, 
        trk,
        mask_positions,
        brain_template,
        v2f_mapping,
        smoothing = False, 
        mode = "roi", 
        verbose = False):
    """
    Docstring for vectorised_probability_maps
    
    :param atlas_path: str
        Path to the file containing the atlas.
    :param reference_file: str
        Path to a reference image for the patient. Should be a nifti file.
        Designed with a T1w file, could be a trk file with affine info.
    :param trk: Tractogram object.
        Tractogram for the subject
    :param remap: Boolean
        Determines if a new mapping is calculated, rather than using a saved 
        one.

    :returns connection_probability
        An array - 
    """

    # Move the tractogram to the corner of the voxel
    #trk.to_corner()

 
    # Extract the overall density map
    overall_density_map = density_map(
        streamlines=trk.streamlines,
        affine=np.eye(4),
        vol_dims=trk.dimensions)
    
    if smoothing:
        overall_density_map = gaussian_filter(
            overall_density_map, 
            1.0
        )
    
    """ Iterate through the atlas regions and extract only the streamlines
        that go through each region. (116 iterations, will give a NxN 
        matrix for each ROI, where N is the number of white matter voxels) """
    if mode == "roi":
        atlas_matrix = registered_atlas.get_fdata()
        roi_ids = np.unique(atlas_matrix) 
        # Stack the density maps up into a single array.
        N = len(roi_ids)-1
        all_density_maps = np.zeros(
            shape=(N,overall_density_map.shape[0], 
                    overall_density_map.shape[1], 
                    overall_density_map.shape[2]))

        for idx, roi in tqdm(enumerate(roi_ids),"\tComputing ROI streamlines"):
            idx -= 1
            if roi == 0:
                continue

            # Get all the streamlines that reach the grey matter ROI
            mask =  (atlas_matrix == roi).astype(np.uint8)

            if mask.shape != atlas_matrix.shape:
                raise ValueError("The mask does not match " \
                    "the shape of the atlas")
            
            relevant_streamlines = target_1(trk = trk, 
                                            mask = mask,
                                            affine = np.eye(4))
            
            trk_new = trk.from_sft(
                relevant_streamlines, 
                trk
            )

            if len(trk_new.streamlines) == 0:
                print(f"Warning: Region {roi} has 0 streamlines")

            if verbose:
                max_coord = np.max(np.vstack(trk_new.streamlines), axis=0)
                min_coord = np.min(np.vstack(trk_new.streamlines), axis=0)
                print("Min:", min_coord)
                print("Max:", max_coord)
                print("Shape:", trk.dimensions)
                
                print(f"The origin is: {trk.origin}\n"
                    f"The space is: {trk.space}\n"
                    f"Dimensions: {trk.dimensions}"
                )
            roi_density_map = density_map(
                streamlines=trk_new.streamlines, 
                affine=np.eye(4), 
                vol_dims=trk.dimensions
            )
            if smoothing:
                roi_density_map = gaussian_filter(roi_density_map, 1.0)
            # Output the density maps so that they can be visualised
            all_density_maps[idx] = roi_density_map

    elif mode == "vox":
        print("\tUsing the Voxel mode")
        brain_template = brain_template.get_fdata()
        all_density_maps = []

        # Need to loop through grey matter voxels:
        failure_count = 0

        for grey_vox, _ in tqdm(
                enumerate(mask_positions), 
                desc="\tVoxel wise fibre connectivity", 
                leave=True):
            
            # Compute the relevant streamlines for that voxel:
            if tuple(mask_positions[grey_vox]) in v2f_mapping.keys():
                streamline_idxs = v2f_mapping[tuple(mask_positions[grey_vox])]
                vox_density_map = density_map(
                    streamlines=trk.streamlines[streamline_idxs], 
                    affine = np.eye(4),
                    vol_dims=trk.dimensions
                    )    
                
            else:
                vox_density_map = np.zeros_like(brain_template)  
                failure_count+=1
               
            sarr = sparse.COO.from_numpy(vox_density_map)
            all_density_maps.append(sarr)

        failure_proportion = failure_count/len(mask_positions)     
        all_density_maps = sparse.stack(all_density_maps, axis=0)                         
        
    else:
        raise ValueError("Please enter a valid mode. " \
                        "Valid modes are: 'roi', 'vox'")
        
    return all_density_maps, overall_density_map

def compute_connection_probability(
        overall_density_map, 
        all_density_maps):
    try:
        safe_overall_density = np.where(
            overall_density_map==0, 
            1, 
            overall_density_map)
        
        connection_probability = all_density_maps / safe_overall_density

    except ValueError:
        all_density_maps = np.transpose(all_density_maps, (3,0,1,2))
        connection_probability = all_density_maps / overall_density_map

    if is_sparse(connection_probability) == False:

        connection_probability = np.nan_to_num(
            connection_probability, 
            True, 
            nan=0)

    return connection_probability

def normalizer(funct_results, 
               probability_maps, 
               method = "basic",
               bold_min = None, 
               bold_max = None):
    """
    Converts the raw functionnectome values to constrain them to
    within the range of the original BOLD signal. Some of these 
    methods are experimental/require validation. 
    
    :param funct_results: Array-like
        Object containing the raw values of the functionnectome.
    :param probability_maps: Array
        Contains the probability maps for each voxel/ROI
    :param method: str
        String defining which method to use. Valid options are 
        ('basic', 'self-sum', 'voxel_sum', 'hack')
    :param bold_min: OPTIONAL Numeric
        The minimum value in the bold data. Only required for 
        the hack method
    :param bold_max: OPTIONAL Numeric
        The maximum value in the bold data. Only required for 
        the hack method.
    """
    if np.sum(np.isnan(funct_results))>0:
        raise ValueError("Unscaled Functionnectome contains NAN values.")

    if method == "basic":
        summed_probs = np.sum(probability_maps) # Dud - makes values minute
    elif method == "self-sum": # Dud - makes values too large
        summed_probs = 0
        for probability_map in probability_maps:
            summed_probs += np.mean(probability_map)
    elif method == "voxel_sum": # Currently a dud - returns entirely NaNs.
        summed_probs = np.nansum(probability_maps, axis=0)

        print(
            f"Sum properties:\n"
            f" Nonzeros: {np.count_nonzero(summed_probs)}\n"
            f" Nansum: {np.nansum(summed_probs, axis=0)}\n"
            f"Original properties:\n"
            f" Non-zeros: {np.count_nonzero(probability_maps)}"
            )

    elif method == "hack": # I think this may sacrifice the 
        #physiological meaning of what we are doing.
        if bold_min == None or bold_max == None:
            raise ValueError("Please enter the parameters bold_min and " \
                            "bold_max (numeric) to use the hack method.")
        max_val = funct_results.max()
        min_val = funct_results.min()
        if max_val-min_val==0:
            raise ValueError("All values are zero. Ensure that the atlas and " \
                "the streamlines overlap")
        normalised = (((funct_results-funct_results.min())
                        / (max_val- min_val)) 
                         * (bold_max-bold_min)
                         + bold_min)
        return normalised
    elif method == "none":
        return funct_results
    else:
        raise ValueError("Please enter a valid method "
                        "('basic', 'self-sum', 'voxel_sum', 'hack')")
    
    normalised = funct_results/summed_probs   
    return normalised
    
def functionnectome(probability_maps, 
                    timeseries, 
                    registered_atlas,
                    normalisation="hack",
                    extensive_visualisation=False, 
                    debug_prints= False):
    """
    Computes the functionnectome based on a probability of 
    connection map and the fmri data.
    
    :param probability_maps: array like
        Contains the probability of connection between a voxel 
        and regions of interest.
    :param fMRI_file: str
        Contains the fMRI BOLD data - at this point this needs 
        to be in the MNI space (which fMRI prep produces as well)
        Maybe no longer as restrictive? 
    :param registered_atlas: str
        Contains the atlas that has been adapted to the patient 
        T1 space.
    """

    if extensive_visualisation != False:
        reg_atlas_img = nifti_vs_img(registered_atlas)
        img_by_ROI = signals_to_img_labels(signals=timeseries,
                                        labels_img=reg_atlas_img)
        img_by_ROI.to_filename(extensive_visualisation)

    bold_max = np.max(timeseries)
    bold_min = np.min(timeseries)

    if debug_prints:
        print(f"Minimum BOLD value: {bold_min}\nMaximum BOLD value: {bold_max}"
              f"Shape of prob_maps: { probability_maps.shape}"
              f"ROI_time series shape: {timeseries.shape}") 
        

    if is_sparse(probability_maps):
        funct_result = sparse.tensordot(timeseries, probability_maps)
    else:
        funct_result = tensordot(timeseries, probability_maps,1) 

    print(f"\tNANs 1 : {np.sum(np.isnan(funct_result))}")

    funct_result = normalizer(
        funct_results=funct_result,
        probability_maps=probability_maps,
        method=normalisation,
        bold_min=bold_min,
        bold_max=bold_max)

    print(f"\tNANs 2: {np.sum(np.isnan(funct_result))}")

    funct_result = np.transpose(funct_result, (1,2,3,0))

    if debug_prints:
        print(f"Max F value: {funct_result.max()}\n"
              f"Min F value: {funct_result.min()}"
              )

    return funct_result

def plot_timeseries(roi_time_series):
    plt.plot(roi_time_series)
    plt.show()

def visual_inspection(density_map_img, region):
    data = density_map_img.get_fdata()
    print(f"Report for {region}\n"
          f"Minimum Value: {np.min(data)}\n"
          f"Maximum Value: {np.max(data)}\n"
          f"Non-zeros    : {np.count_nonzero(data)}"
          )
  
def plot_ROI_activity(registered_atlas, roi_timeseries):
    """
    Goal is to prepare a visualisation that contains the bold 
    signal in each ROI 
    
    :param registered_atlas: Description
    :param roi_timeseries: Description
    """
    atlas_img = nib.load(registered_atlas)
    atlas_data = atlas_img.get_fdata()
    image_data = np.zeros(
        shape=(roi_timeseries.shape[0], 
               roi_timeseries.shape[1], 
               atlas_data.shape[0], 
               atlas_data.shape[1], 
               atlas_data.shape[2]))
    
    print("Shape is", image_data.shape)

    for idx in tqdm(range(roi_timeseries.shape[1]), "Preparing ROI Images"):
        for j in range(roi_timeseries.shape[0]):
            roi = idx + 1
            image_data[j, idx] = np.where(registered_atlas==roi, 
                                          roi_timeseries[j, idx], 
                                          np.nan)


"""
Possibly useful detritus - maybe need to add some of these to the pipeline

Originally from the probability maps vectorised. It would check
if the values were saved and if they were, it would load them

    if save_density_map_path != None:
        filepath = path.join(save_density_map_path, 
                             f"{mode}_density_map.nii.gz")

        if os.path.exists(filepath):
                print("\tLoading Pre-existing density maps")
                all_density_maps = nib.load(filepath)
                all_density_maps = all_density_maps.get_fdata()
                return all_density_maps, overall_density_map

        if save_density_map_path != None:
            complete_density_map_data = np.transpose(
                    all_density_maps, 
                    (1,2,3,0))
            out = nib.Nifti1Image(
                    complete_density_map_data.astype(float), 
                    trk.affine)
            out.to_filename(filename=filepath)

    if save_output != None and is_sparse(connection_probability) == False:
        if type(save_output) is not str:
           raise ValueError("Please ensure save_output is a string " \
                            "filepath to save the probability maps")
        
        save_name = path.join(save_output, f"probability_maps.nii.gz")
        save_values = np.transpose(connection_probability, (1,2,3,0))
        out = nib.Nifti1Image(save_values.astype(float), trk.affine)
        out.to_filename(save_name)



"""