import numpy as np
import nibabel as nib
from fmri_processing import connectivity_matrix_generation
from nilearn import image
from networkx import edge_betweenness_centrality, Graph
import networkx as nx
import matplotlib.pyplot as plt
from nilearn.plotting import plot_matrix, show
from utilities import voxel_to_streamline_map
from dipy.io.streamline import load_tractogram
from utilities import mask_generator, generate_masks
from tqdm import tqdm
from unravel.analysis import connectivity_matrix
import sparse
from dipy.io.stateful_tractogram import Origin, Space
import os.path as path


def engagement_pipeline(bold_data, atlas, tractogram_file, white_matter_prob, cache_pathway = None, plotting = False, save_engagement = None, verbose = False, grey_matter_prob = None,  csf_prob = None):
    """
    Pipeline that performs the entire engagement calculation - functions within this will correspond to submodules 
    that can be run with just the required objects. This function works with the filepaths.
    
    :param bold_data: str
        Preprocessed bold data. Can be in any space (T1w, MNI), however this must match the space of the atlas.
    :param atlas: str
        Filepath to the atlas. Space must match the bold data space
    :param tractogram_file: str
        Filepath for the tractogram
    :param white_matter_prob: str
        Filepath for the white matter probability map.
    :param plotting: boolean
        Default is False. If true, the correlation matrix will be plotted.
    :param save_engagement: str
        A filepath to save the engagement results.
    :param verbose: Boolean
        If true, more printing will occur.
    :param grey_matter_prob: Optional (unused as of right now)
    :param csf_prob: Optional (unused as of right now)
    """
    task = 1
    ################################ Step 1 ################################
    print(f"{task}. Preparing functional connectivity matrix")
    # Load the bold data and the atlas

    bold_img = image.load_img(bold_data)
    bold_img_data = bold_img.get_fdata()
    bold_img_data = bold_img_data[:, :, :, 3:]
    atlas_img = nib.load(atlas)
    atlas_data = atlas_img.get_fdata()

    atlas_values = np.unique(atlas_data)
    
    ROIs = len(atlas_values)

    # Check if the functional connectivity matrix already exists
    if cache_pathway is not None:
        fc_path = path.join(cache_pathway, "fc_matrix.npy")
        if path.exists(fc_path):
            fc_mat = np.load(fc_path)
        else:
            fc_mat = connectivity_matrix_generation(bold_img, atlas, False)
    else:
        fc_mat = connectivity_matrix_generation(bold_img, atlas, False)
    
    
    if verbose:
        print("The atlas looks like: ", atlas_data)
        print("Atlas values are", atlas_values)
        print("The FC matrix looks like: ", fc_mat)

    # Plotting to see if the matrix makes sense (as of right now it does not!!!)
    if plotting:
        np.fill_diagonal(fc_mat, 0)
        plot_matrix(
        fc_mat,
        labels = np.unique(atlas_img.get_fdata())[np.unique(atlas_img.get_fdata())!=0],
        figure=(10, 8),
        vmax=0.8,
        vmin=-0.8,
        title="Raw correlations",
        reorder=True,
        )
        plt.show()

    # Threshold the correlations to make it amenable to the EBC metric (may make sense to replace 
    # this with a metric that more accurately characterises the degree of "proximity" a node has to other nodes")
    fc_mat = correlation_thresholding(fc_mat, 1.0)

    if verbose:
        print("After thresholding: ", fc_mat)

    task+=1

    ################################ Step 2 ################################
    print(f"{task}. Computing Edge Between Connectedness Matrix") 
    ebc_mat = ebc_computation(fc_mat)
    if verbose:
        print("The ebc matrix")
        print(ebc_mat)
        print(ebc_mat.shape)
    task += 1

    ################################ Step 3 ################################
    print(f"{task}. Computing all fibres that penetrate each voxel")   

    trk = load_tractogram(tractogram_file, "same")

    all_connectivity_matrices, wm_positions = generate_VWSC_matrices(white_matter_prob=white_matter_prob,
                                                                     trk=trk,
                                                                     atlas_data=atlas_data)

    print("The connectivity matrices:")
    print(all_connectivity_matrices)

    all_connectivity_matrices = sparse.asnumpy(all_connectivity_matrices)
    print(all_connectivity_matrices)

    task += 1
    ################################ Step 4 ################################
    print(f"{task}. Calculating Engagement")   
    engagement = engagement_calculation(EBC_matrix=ebc_mat,
                           SC_matrices=all_connectivity_matrices)
    
    print("The final result")
    print(engagement)
    print(engagement.min(), engagement.max())
    task += 1

    ################################ Step 5 ################################
    if save_engagement!= None:
        print(f"{task}. Saving Result")
        print(engagement.shape)
        engagement = sparse.asnumpy(engagement)
        save_engagement(engagement, 
                        wm_positions, 
                        atlas_img.dimensions, 
                        save_engagement, 
                        atlas_img.affine)
    
    print("Finito!")
    

def save_engagement(engagement_values, wm_positions, dimensions, save_path, affine):
    """
    Save the results of the engagement calculation.
    
    :param engagement_values: array
        A numpy array containing engagement values. Must be single indexed. Indexing should align with the white matter masks used to generate the engagement scores.
    :param wm_positions: list
        A list containing triple values that are indices to a white matter voxel in the original brain scan. 
    :param dimensions: 3 tuple
        The dimensions of the original brain scan. This is used to cast the engagement scores back to the correct shape.
    :param save_path: Str
        Filepath to save the engagement values.
    :param affine: np.array 4x4
        The affine information used to create a new nifti image. Must align with the affine information from the original brain scans used to create the engagement metrics.
    """
    # Create a blank brain to hold the values
    brain_template = np.zeros(shape = dimensions)
    for idx, position in enumerate(tqdm(wm_positions, "Reshaping Engagement")):
        brain_template[position] = engagement_values[idx]
    
    out = nib.Nifti1Image(brain_template, affine)
    out.to_filename(save_path)



def engagement_calculation(EBC_matrix, SC_matrices):
    result = sparse.einsum("ijk,jk->i", SC_matrices, EBC_matrix)
    denom = SC_matrices.sum(axis=(1, 2)) # This line converts each slice of the SC_matrices array into a single number (the sum of all the values in that slice)
    result = np.where(result != 0, result / denom, result) # This line performs the division where the denom is non-zero. Otherwise leave as is.

    return result


def trk_report(trk, value):
    """
    Simple function to report the relevant properties of the tractograms that are passed to key functions. Only used for debugging.
    
    :param trk: Stateful Tractogram
        Tractogram of interest.
    :param value: Int or Str
        Simple flag to identify which point in the code is generating the report.
    """
    print(f"TRK Status Check {value}")
    print(f"Affine {trk.affine}\nDimensions:{trk.dimensions}\nOrigin: {trk.origin}\nSpace: {trk.space}")
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


def generate_VWSC_matrices(atlas_data, trk, white_matter_prob = None, white_matter_mask = None, verbose = False):
    """
    Generate a structural connectivity matrix for every white matter voxel. It first generates a mapping of voxel to streamline. 
    This identifies the subset of streamlines that pass through the voxel. Then, it generates a white matter mask based on provided probability maps.
    The positions of each white matter voxel are then extracted from the mask. For each voxel, a connectivity matrix is generated showing how 
    strongly each region of interest is connected via the voxel. These are stored as sparse arrays and returned as a sparse array.
    
    :param atlas_data: Array like
        The labels of the ROI. 
    :param trk: Stateful_Tractogram
        Tractogram containing all streamlines for a patient
    :param white_matter_prob: str/Nifti image
        Provides the probabilities for each voxel being white matter. Provide either white_matter_probs or white_matter_mask
    :param white_matter_mask: tr/Nifti image
        A white matter mask. Provide either white_matter_probs or white_matter_mask
    :param verbose: If true, prints the number or failures. 
    """
    if white_matter_mask is None and white_matter_prob is None:
        raise ValueError("Please provide either a white_matter_mask or a white_matter_probability file")
    
    # Functions as a good check to ensure that everything is in the right space. 
    #trk_report(trk, 1)

    # Ensure that the coordinates are in voxel space and in the corner (vistrack representation)
    if trk.space != Space.VOX:
        trk.to_vox()
    if trk.origin != Origin.TRACKVIS:
        trk.to_corner()

    #trk_report(trk, 2)
    v2f_mapping = voxel_to_streamline_map(trk.streamlines, vol_shape=trk.dimensions)


    # Generate a white matter mask if probability is provided:
    if white_matter_mask == None:
        wm_mask = mask_generator(white_matter_probability=white_matter_prob, smoothing=False)
    else:
        wm_mask = nib.load(white_matter_mask)
    
    # Generate all white matter positions
    wm_positions = generate_masks(wm_mask)

    # Naive method:
    all_connectivity_matrices = []


    path_1_count = 0 
    non_zero_count = 0
    ROIs = len(np.unique(atlas_data))
    
    for idx, voxel in enumerate(tqdm(wm_positions, "VW SC matrices")):

        if tuple(voxel) not in v2f_mapping.keys():
            conn_mat = np.zeros(shape=(ROIs, ROIs))
            path_1_count +=1
        else:
            streamline_indices = v2f_mapping[tuple(voxel)]
            conn_mat = connectivity_matrix(trk.streamlines[streamline_indices], atlas_data,inclusive=False)
            
        
        conn_mat = np.delete(conn_mat, 0, 0)
        conn_mat = np.delete(conn_mat, 0, 1)

        if np.count_nonzero(conn_mat) > 0:
                non_zero_count += 1

        conn_mat = sparse.COO.from_numpy(conn_mat)
        all_connectivity_matrices.append(conn_mat)

    if verbose:
        print(f"The number of voxels with no streamlines: {path_1_count} out of {len(wm_positions)}")
        print(f"The number of voxels  connectivity matrices with at least one connection: {non_zero_count} out of {len(wm_positions)}")


    
    all_connectivity_matrices = sparse.stack(all_connectivity_matrices, axis = 0)
    return all_connectivity_matrices, wm_positions

def ebc_computation(numpy_matrix):
    """
    Simple wrapper to calculate the EBC matrix starting with a functional connectivity matrix.
    
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


def correlation_thresholding(matrix, proportion=0.9, keep_diagonal=False):
    """
    Threshold a functional connectivity array, only retaining values that are in the top 0.x of the data.
    
    :param matrix: Array
        Numpy array
    :param proportion: numeric
        Decimal proportion of data to keep. Range between 0 and 1
    """
    if not 0 <= proportion <= 1:
        raise ValueError("Ensure proportion is between 0 and 1")

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

    if keep_diagonal:
        np.fill_diagonal(filtered, np.diag(matrix))

    return filtered

