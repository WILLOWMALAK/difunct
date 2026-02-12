import os.path as path
import os
import numpy as np
import sparse
import nibabel as nib
from nilearn import image
from nilearn.plotting import plot_matrix, show
import matplotlib.pyplot as plt
from networkx import edge_betweenness_centrality, Graph
import networkx as nx
from dipy.io.stateful_tractogram import Origin, Space
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import select_by_rois
from tqdm import tqdm
from unravel.analysis import connectivity_matrix
import matplotlib.pyplot as plt
from utilities import connectivity_matrix_generation,mask_generator, mask_to_positions
from utilities import voxel_to_streamline_map, voxel_to_streamline_map_V2  
from utilities import create_ROI_time_series, fc_mat_gen, nifti_vs_img, is_sparse, sl_to_roi_map
from utilities import conn_matrices, conn_matrices_V2



def save_engagement(engagement_values, 
                    wm_positions, 
                    dimensions, 
                    save_path, 
                    affine):
    """
    Save the results of the engagement calculation.
    
    :param engagement_values: array
        A numpy array containing engagement values. Must be single indexed. 
        Indexing should align with the white matter masks used to generate 
        the engagement scores.
    :param wm_positions: list
        A list containing triple values that are indices to a white matter 
        voxel in the original brain scan. 
    :param dimensions: 3 tuple
        The dimensions of the original brain scan. This is used to cast the 
        engagement scores back to the correct shape.
    :param save_path: Str
        Filepath to save the engagement values.
    :param affine: np.array 4x4
        The affine information used to create a new nifti image. Must align with
          the affine information from the original brain scans used to create 
          the engagement metrics.
    """
    brain_template = reshape_engagement(shape=dimensions, 
                                        wm_positions=wm_positions, 
                                        engagement_vals=engagement_values)

    out = nib.Nifti1Image(brain_template, affine)
    out.to_filename(save_path)

def matrix_report(matrix):
    print("Matrix Report")
    print(f"Max: {matrix.max()}")
    print(f"Min: {matrix.min()}")
    print(f"Nonzeros: {np.count_nonzero(matrix)}")

def engagement_calculation(EBC_matrix, 
                           SC_matrices, 
                           method = "einsum"):
    """
    Computes engagement from an edge between connectivity matrix and a 
    structural connectivity matrix.
    
    :param EBC_matrix:  Array
        NxN matrix that contains the edge between connectedness value for each 
        edge in the functional connectome. N is the number of regiosn of
        interest. 
    :param SC_matrices: Array
        MxNxN. An array of connectomes, containing the connectivity of each 
        region of interest through the voxel m. 
    :param method: Str
        Either 'einsum' or 'custom'. These are two realisations - one more 
        explicit using loops and one more efficient. Default is einsum and 
        this is recommended for efficiency.
    """
    if method == "einsum":
        try:
            result = sparse.einsum("ijk,jk->i", SC_matrices, EBC_matrix)
        except ValueError as e:
            print(f"Dimensions of SC_matrices: {SC_matrices.shape}"
                  f"Dimensions of EBC_matrix: {EBC_matrix.shape}")
        denom = SC_matrices.sum(axis=(1, 2))# This line converts each slice of 
        #the SC_matrices array into a single number (the sum of all the values 
        #in that slice)
        result = np.where(denom != 0, result / denom, 0)
    elif method == "custom":
        result = []
        numerators = []
        denominators = []

        for SC_matrix in SC_matrices:
            numerator = sparse.sum(sparse.multiply(SC_matrix, EBC_matrix))
            numerators.append(numerator)
            denom = sparse.sum(SC_matrix)
            if denom != 0:
                result.append(numerator/denom)
            else:
                if numerator == 0:
                    result.append(0)
                else:
                    raise ValueError(f"The denominator is 0 "
                                     f"but the numerator is {numerator}")
            denominators.append(denom)
            
        result = np.array(result)

        plt.hist(numerators)
        plt.title("numerator distribution")
        plt.show()
        plt.hist(denominators)
        plt.title("Denominators")
        plt.show()
        return result
    else:
        raise ValueError(f"{method} is not a valid method")
    
    return result

def trk_report(trk, value):
    """
    Simple function to report the relevant properties of the tractograms that 
    are passed to key functions. Only used for debugging.
    
    :param trk: Stateful Tractogram
        Tractogram of interest.
    :param value: Int or Str
        Simple flag to identify which point in the code is generating the 
        report.
    """
    print(f"TRK Status Check {value}")
    print(f"Affine {trk.affine}\nDimensions:{trk.dimensions}\n")
    print(f"Origin:{trk.origin}\nSpace: {trk.space}")

    streamline_obj = list(trk.streamlines)

    max_ax0 = -10000
    min_ax0 = 10000
    max_ax1 = -10000
    min_ax1 = 10000
    max_ax2 = -10000
    min_ax2 = 10000

    for streamline in streamline_obj:
        max_ax0_sl = streamline[:, 0].max()
        min_ax0_sl = streamline[:, 0].min()
        max_ax1_sl = streamline[:, 1].max()
        min_ax1_sl = streamline[:, 1].min()
        max_ax2_sl = streamline[:, 2].max()
        min_ax2_sl = streamline[:, 2].min()

        if max_ax0_sl > max_ax0:
            max_ax0 = max_ax0_sl
        if min_ax0_sl < min_ax0:
            min_ax0 = min_ax0_sl
        if max_ax1_sl > max_ax1:
            max_ax1 = max_ax1_sl
        if min_ax1_sl < min_ax1:
            min_ax1= min_ax1_sl
        if max_ax2_sl > max_ax2:
            max_ax2 = max_ax2_sl
        if min_ax2_sl < min_ax2:
            min_ax2= min_ax2_sl



    print(f"Axis 0 Max = {max_ax0}, Min = {min_ax0}")
    print(f"Axis 1 Max = {max_ax1}, Min = {min_ax1}")
    print(f"Axis 2 Max = {max_ax2}, Min = {min_ax2}")

def generate_VWSC_matrices_entire_sl(
        atlas_data, 
        trk, 
        v2f_mapping = None,
        white_matter_prob = None, 
        white_matter_mask = None, 
        segmentation = 1,   
        sift2_weights = None,
        sift2_mu = None):
    """
    Generate a structural connectivity matrix for every white matter voxel. 
    It first generates a mapping of voxel to streamline. This identifies the 
    subset of streamlines that pass through the voxel. Then, it generates a 
    white matter mask based on provided probability maps. The positions of each 
    white matter voxel are then extracted from the mask. For each voxel, a 
    connectivity matrix is generated showing how strongly each region of 
    interest is connected via the voxel. These are stored as sparse arrays and 
    returned as a sparse array.
    
    :param atlas_data: Array like
        The labels of the ROI. 
    :param trk: Stateful_Tractogram
        Tractogram containing all streamlines for a patient
    :param white_matter_prob: str/Nifti image
        Provides the probabilities for each voxel being white matter. Provide 
        either white_matter_probs or white_matter_mask
    :param white_matter_mask: tr/Nifti image
        A white matter mask. Provide either white_matter_probs or 
        white_matter_mask
    :param verbose: If true, prints the number or failures. 
    """
    if white_matter_mask is None and white_matter_prob is None:
        raise ValueError(f"Please provide either white_matter_mask" 
                         f"or white_matter_probability file")
    
    if trk.space != Space.VOX:
        trk.to_vox()
    if trk.origin != Origin.TRACKVIS:
        trk.to_corner()

    if v2f_mapping is None:
        v2f_mapping = voxel_to_streamline_map_V2(
            trk.streamlines, 
            vol_shape=trk.dimensions,
            subsegment=segmentation)

    non_empty = 0
    for voxel in v2f_mapping.keys():
        if len(v2f_mapping[voxel]) != 0:
            non_empty += 1
    if non_empty == 0:
        raise ValueError("The mapping identified no " \
                        "voxels containing streamlines")

    # Generate a white matter mask if probability is provided:
    if white_matter_mask is None:
        wm_mask = mask_generator(white_matter_probability=white_matter_prob, 
                                 smoothing=False)
    else:
        wm_mask = nib.load(white_matter_mask)
    wm_positions = mask_to_positions(wm_mask)
    sl_roi_map  = sl_to_roi_map(
        trk.streamlines,
        atlas_data,
        only_endpoints=False
    )
    all_connectivity_matrices = conn_matrices_V2(
        sl_roi_map=sl_roi_map,
        vox_sl_map=v2f_mapping,
        atlas_data=atlas_data,
        mask_positions=wm_positions,
        sift_2_weights=sift2_weights,
        sift_2_mu=sift2_mu
    )
    return all_connectivity_matrices, wm_positions


def generate_VWSC_matrices_ep_only(
        atlas_data, 
        trk, 
        v2f_mapping = None,
        white_matter_prob = None, 
        white_matter_mask = None, 
        sift2_weights = None,
        sift2_mu = None,
        segmentation = 1):
    """
    Generate a structural connectivity matrix for every white matter voxel. 
    It first generates a mapping of voxel to streamline. This identifies the 
    subset of streamlines that pass through the voxel. Then, it generates a 
    white matter mask based on provided probability maps. The positions of each 
    white matter voxel are then extracted from the mask. For each voxel, a 
    connectivity matrix is generated showing how strongly each region of 
    interest is connected via the voxel. These are stored as sparse arrays and 
    returned as a sparse array.Endpoints only.
    
    :param atlas_data: Array like
        The labels of the ROI. 
    :param trk: Stateful_Tractogram
        Tractogram containing all streamlines for a patient
    :param white_matter_prob: str/Nifti image
        Provides the probabilities for each voxel being white matter. Provide 
        either white_matter_probs or white_matter_mask
    :param white_matter_mask: tr/Nifti image
        A white matter mask. Provide either white_matter_probs or 
        white_matter_mask
    :param verbose: If true, prints the number or failures. 
    """
    if (white_matter_mask is None 
        and white_matter_prob is None):
        raise ValueError(f"Please provide either white_matter_mask" 
                         f"or white_matter_probability file")
    
    if trk.space != Space.VOX:
        trk.to_vox()
    if trk.origin != Origin.TRACKVIS:
        trk.to_corner()
    if v2f_mapping is None:
        v2f_mapping = voxel_to_streamline_map_V2(
            trk.streamlines, 
            vol_shape=trk.dimensions,
            subsegment=segmentation)

    non_empty = 0

    for voxel in v2f_mapping.keys():
        if len(v2f_mapping[voxel]) != 0:
            non_empty += 1
    if non_empty == 0:
        raise ValueError("The mapping identified no " \
                        "voxels containing streamlines")

    # Generate a white matter mask if probability is provided:
    if white_matter_mask is None:
        wm_mask = mask_generator(
            white_matter_probability=white_matter_prob, 
            smoothing=False)
    else:
        wm_mask = nib.load(white_matter_mask)
    wm_positions = mask_to_positions(wm_mask)
    sl_roi_map  = sl_to_roi_map(
        trk.streamlines,
        atlas_data
    )
    all_connectivity_matrices = conn_matrices_V2(
        sl_roi_map=sl_roi_map,
        vox_sl_map=v2f_mapping,
        atlas_data=atlas_data,
        mask_positions=wm_positions,
        sift_2_weights=sift2_weights,
        sift_2_mu=sift2_mu
    )
    return all_connectivity_matrices, wm_positions


def ebc_computation(numpy_matrix, inverted_values):
    """
    Simple wrapper to calculate the EBC matrix starting with a functional 
    connectivity matrix.
    
    :param numpy_matrix: np array
        Functional connectivity array.
    """
    if inverted_values:
        numpy_matrix = 1/numpy_matrix

    g = nx.from_numpy_array(numpy_matrix, 
                            edge_attr = "weight")
    
    ebc_dict= edge_betweenness_centrality(
        G=g, 
        weight="weight", 
        normalized=False
    )
    ebc_mat = np.zeros_like(numpy_matrix)
    for key in ebc_dict.keys():
        ebc_mat[key[0], key[1]] = ebc_dict[key]
        ebc_mat[key[1], key[0]] = ebc_dict[key]
    return ebc_mat

def matrix_value_thresholding(matrix, value_threshold = 0.2):
    """
    Produces a new matrix, retaining only values over a certain threshold. 
    Default is 0.2, which applies mainly to correlation matrices. 
    
    :param matrix: Description
    :param value_threshold: Description
    """
    if value_threshold >= 0: 
        filtered = np.where(matrix > value_threshold, matrix, 0)
    else:
        filtered = np.where(matrix < value_threshold, matrix, 0)
    return filtered

def correlation_thresholding(matrix, proportion=0.9, 
                             keep_diagonal=False, 
                             remove_negatives = True, 
                             value_threshold = None):
    """
    Two behaviours encoded in one. 
    
    :param matrix: Array
        Numpy array
    :param proportion: numeric
        Decimal proportion of data to keep. Range (0, 1]

    """
    if value_threshold is not None:
        filtered = matrix_value_thresholding(
            matrix,
            value_threshold)
        if keep_diagonal:
            np.fill_diagonal(filtered, np.diag(matrix))
        else:
            np.fill_diagonal(filtered, 0)
        return filtered
    
    if not 0 < proportion <= 1:
        raise ValueError("Ensure proportion is greater than 0 and less than or equal to 1")

    n = matrix.shape[0]

    # Upper triangle indices (unique edges)
    iu = np.triu_indices(n, k=1)
    values = matrix[iu]

    # Number of edges to keep
    k = int(np.ceil(proportion * values.size))
    if k == 0:
        return np.zeros_like(matrix)

    # Find cutoff by rank (not value quantile)
    cutoff = np.partition(values, -k)[-k]

    # Create boolean mask on upper triangle
    mask_ut = values >= cutoff

    # Initialize output
    filtered = np.zeros_like(matrix)

    # Assign kept edges symmetrically
    filtered[iu[0][mask_ut], iu[1][mask_ut]] = values[mask_ut]
    filtered[iu[1][mask_ut], iu[0][mask_ut]] = values[mask_ut]

    # Remove any negative values and replace with zero
    if remove_negatives:
        filtered = np.where(filtered<0, 0, filtered)
    if keep_diagonal:
        np.fill_diagonal(filtered, np.diag(matrix))

    return filtered

def save_connectivity_matrices(all_connectivity_mats, save_path):
    np.save(file=save_path,
            arr=all_connectivity_mats)

def dynamic_engagement(sliced_time_series, connectivity_matrices):
    ebc_matrices = []

    for slice in tqdm(sliced_time_series, "Slicewise EBC"):
        fc_matrix  = fc_mat_gen(timeseries=slice)
        ebc = ebc_computation(fc_matrix, False)
        ebc_matrices.append(ebc)

    ebc_matrices = np.stack(ebc_matrices)

    if is_sparse(connectivity_matrices):
        connectivity_matrices = sparse.asnumpy(connectivity_matrices)


    engagement = np.einsum(
        "ijk,njk->ni", 
        ebc_matrices, 
        connectivity_matrices)

    return engagement

def reshape_engagement(
        shape, 
        wm_positions, 
        engagement_vals
):
    if len(wm_positions) != len(engagement_vals):
        raise ValueError("The wm_positions and engagement values" \
        " are not the same size")
    
    brain_template = np.zeros(shape = shape)
    for idx, position in enumerate(wm_positions):
        try:
            value = engagement_vals[idx]
            brain_template[tuple(position)] = value
        except IndexError as e:
            print(f"Index: {idx}, Position: {position}\n"
                  f"Max allowed index: {len(engagement_vals)-1}\n"
                  f"Length of wm mask: {len(wm_positions)}")
            raise e
            

    return brain_template

def reshape_engagement_slices(
        sliced_engagement, 
        image_template, 
        wm_positions):
    
    all_slices = []
    for i in tqdm(range(sliced_engagement.shape[1]), "Reshaping"):
        slice =  reshape_engagement(
            shape=image_template.shape,
            wm_positions=wm_positions,
            engagement_vals=sliced_engagement[:, i]
        )
        all_slices.append(slice)

    return np.stack(all_slices, axis=-1)
