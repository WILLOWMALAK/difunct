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
from tqdm import tqdm
from unravel.analysis import connectivity_matrix
from utilities import connectivity_matrix_generation,mask_generator, generate_masks
from utilities import voxel_to_streamline_map  

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
    # Create a blank brain to hold the values
    brain_template = np.zeros(shape = dimensions)
    for idx, position in enumerate(tqdm(wm_positions, "Reshaping Engagement")):
        print(f"Position: {position}\nValue: {engagement_values[idx]}\n")
        if engagement_values[idx] == 0:
            value = 0
        else:
            value = engagement_values[idx]
        brain_template[tuple(position)] = value

    print()

    out = nib.Nifti1Image(brain_template, affine)
    out.to_filename(save_path)



def engagement_calculation(EBC_matrix, SC_matrices, method = "einsum"):
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
        result = sparse.einsum("ijk,jk->i", SC_matrices, EBC_matrix)
        denom = SC_matrices.sum(axis=(1, 2))# This line converts each slice of 
        #the SC_matrices array into a single number (the sum of all the values 
        #in that slice)
        result_1 = np.where(result != 0, result / denom, result)
    elif method == "custom":
        result = []
        for SC_matrix in SC_matrices:
            numerator = sparse.sum(sparse.multiply(SC_matrix, EBC_matrix))
            denom = sparse.sum(SC_matrix)
            result.append(numerator/denom)
            
        result = np.array(result)

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


def generate_VWSC_matrices(atlas_data, 
                           trk, 
                           white_matter_prob = None, 
                           white_matter_mask = None, 
                           verbose = False):
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
        raise ValueError("Please provide either white_matter_mask or white_matter_probability file")
    
    # Functions as a good check to ensure that everything is in the right space. 
    #trk_report(trk, 1)

    # Ensure that the coordinates are in voxel space and in the corner 
    # (vistrack representation)
    if trk.space != Space.VOX:
        trk.to_vox()
    if trk.origin != Origin.TRACKVIS:
        trk.to_corner()

    #trk_report(trk, 2)
    v2f_mapping = voxel_to_streamline_map(trk.streamlines, 
                                          vol_shape=trk.dimensions)

    print("streamlines", v2f_mapping[(109, 129, 128)])


    # Generate a white matter mask if probability is provided:
    if white_matter_mask == None:
        wm_mask = mask_generator(white_matter_probability=white_matter_prob, 
                                 smoothing=False)
    else:
        wm_mask = nib.load(white_matter_mask)
    
    # Generate all white matter positions
    wm_positions = generate_masks(wm_mask)

    print("white matter positions", wm_positions.shape)

    # Naive method:
    all_connectivity_matrices = []


    path_1_count = 0 
    non_zero_count = 0
    ROIs = len(np.unique(atlas_data))

    no_streamlines = []
    
    for idx, voxel in enumerate(tqdm(wm_positions, "VW SC matrices")):

        if tuple(voxel) not in v2f_mapping.keys():
            conn_mat = np.zeros(shape=(ROIs, ROIs))
            path_1_count +=1
            no_streamlines.append(tuple(voxel))
        else:
            streamline_indices = v2f_mapping[tuple(voxel)]
            conn_mat = connectivity_matrix(trk.streamlines[streamline_indices], 
                                           atlas_data,inclusive=False)
            
        
        conn_mat = np.delete(conn_mat, 0, 0)
        conn_mat = np.delete(conn_mat, 0, 1)

        if np.count_nonzero(conn_mat) > 0:
                non_zero_count += 1

        conn_mat = sparse.COO.from_numpy(conn_mat)
        all_connectivity_matrices.append(conn_mat)

    if verbose:
        print(f"No. of voxels with no streamlines: {path_1_count} out of {len(wm_positions)}")
        print(f"The number of voxel CMs with at least one connection: {non_zero_count} out of {len(wm_positions)}")
        print("The voxels with no streamlines: ")
        print(no_streamlines)

    
    all_connectivity_matrices = sparse.stack(all_connectivity_matrices, axis = 0)
    return all_connectivity_matrices, wm_positions

def ebc_computation(numpy_matrix):
    """
    Simple wrapper to calculate the EBC matrix starting with a functional 
    connectivity matrix.
    
    :param numpy_matrix: np array
        Functional connectivity array.
    """
    g = nx.from_numpy_array(numpy_matrix, 
                            edge_attr = "weight")
    
    ebc_dict= edge_betweenness_centrality(G=g, weight="weight")
    ebc_mat = np.zeros_like(numpy_matrix)
    for key in ebc_dict.keys():
        ebc_mat[key[0], key[1]] = ebc_dict[key]
        ebc_mat[key[1], key[0]] = ebc_dict[key]
    

    return ebc_mat


def value_threshold(matrix, value_threshold = 0.2):
    """
    Produces a new matrix, retaining only values over a certain threshold. 
    Default is 0.2, which applies mainly to correlation matrices. 
    
    :param matrix: Description
    :param value_threshold: Description
    """
    filtered = np.where(matrix > value_threshold, matrix, 0)
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
        filtered = np.where(matrix > value_threshold, matrix, 0)
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


def engagement_feeder(subject_bids_root, subj_id_length, save_root_folder):

    os.makedirs(save_root_folder, exist_ok=True)
    # Make a folder within the subject path
    engagement_storage_path = path.join(subject_bids_root, "engagement")
    os.makedirs(engagement_storage_path, exist_ok=True)
    subj_id = subject_bids_root[-subj_id_length:]
    # Iterate through directory to identify sessions
    for directory in os.listdir(subject_bids_root):
        if directory.__contains__("ses"):
            #Check if there is a functional folder for that session
            func_path = path.join(subject_bids_root, directory, "func")
            if path.exists(func_path):
                bold_filepath = path.join(func_path, subj_id + "_" + directory + "_rest_space-T1w_desc-preproc_bold.nii.gz" )
                atlas_filepath = []