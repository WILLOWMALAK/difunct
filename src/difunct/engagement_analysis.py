import time
import os
from os.path import join

import pandas as pd
import numpy as np
import nibabel as nib
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
from unravel.stream import get_streamline_density, align_streamline, _smooth_streamline
from unravel.stream import get_roi_sections_from_nodes
from tqdm import tqdm
from regis.core import find_transform, apply_transform
from dipy.tracking.streamline import transform_streamlines
from sklearn.decomposition import NMF
from utilities import streamline_registration, trk2tck, nifti_vs_img

MNI_PATH_MINE = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/Atlas_Maps/MNI152_T1_1mm_brain.nii.gz"
MNI_PATH = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/mni_icbm152_nlin_sym_09a_nifti/brain_only.nii.gz"
MNI_MASK = "/Users/sam/Downloads/mni_icbm152_nlin_sym_09a_nifti/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a_mask.nii"
ATLAS_FOLDER = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/Atlas_Maps/Atlas_80_Bundles/Atlas_80_Bundles/bundles"
T1_ANAT_FILE = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/derivatives/sub-TAU001/anat/sub-TAU001_desc-preproc_T1w_brain_only.nii.gz"
OUT_FOLDER = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/Engagement_analysis"
NEW_ATLAS_FP = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/Atlas_Maps/Atlas_80_Bundles/Atlas_80_Bundles/corrected_bundles_V2"
OUR_MNI_BUNDLES =  "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/Atlas_Maps/Atlas_80_Bundles/Atlas_80_Bundles/our_mni_bundles"
PATIENT_FOLDER =  "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/derivatives/sub-TAU001/wm_atlas"
TEST =  "/Users/sam/Desktop/transform_mat.npy"
ENG_STORAGE = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/engagement_anal"
PATIENT_ENG = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/Outputs/TAU001/ses-2/pos_eng.nii.gz"
WITH_INVERTED = "/Users/sam/Documents/sams_pc/University/2025_Univ/Belgium/data_temp/TestFileStructure/derivatives/sub-TAU001/wm_atlas_inverted"

def extract_nodes(trk_file: str, nodes: int = 32, smooth_iter: int = 10):
    '''
    Return the streamline with the most density, subsampled into a defined
    number of nodes.

    Parameters
    ----------
    trk_file : str
        Path to tractogram file
    nodes : int, optional
        Number of points in the average pathway. The default is 32.
    smooth_iter : int, optional
        Number of iterations for smoothing the average pathway. Not recommended
        when there are few nodes. The default is 10.

    Returns
    -------
    centroid : 2D array of size (n, 3)
        Coordinates (x,y,z) of the n mean trajectory points.

    '''

    trk = load_tractogram(trk_file, 'same')
    trk.to_vox()
    trk.to_corner()
    dens = get_streamline_density(trk, subsegment=5)

    streams = trk.streamlines
    s_dens = np.empty(len(streams))
    for i in tqdm(range(len(streams)), desc="Computing centroid streamline"):

        coords = np.floor(streams[i]).astype(np.int32)
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]
        s_dens[i] = np.sum(dens[x,y,z],
                           dtype=np.float32)
    
    s_i_max = np.argmax(s_dens)

    deltas = np.diff(streams[s_i_max], axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    cumulative_dist = np.concatenate(([0], np.cumsum(distances)),
                                     dtype=np.float32)
    new_distances = np.linspace(0, cumulative_dist[-1], nodes, dtype=np.float32)
    indices = np.searchsorted(cumulative_dist, new_distances)
    centroid = streams[s_i_max][indices]

    centroid = align_streamline(centroid)
    if smooth_iter > 0:
        centroid = _smooth_streamline(centroid, iterations=smooth_iter)

    return centroid

def tract_engagement(
        tract_file,
        engagement_file, 
        nodes = 30
):
    trk = load_tractogram(tract_file, reference="same",bbox_valid_check=True)
    if trk is False:
        raise ValueError("Could not load trk")
    eng_img = nifti_vs_img(engagement_file)
    if not np.allclose(trk.affine, eng_img.affine):
        raise ValueError("The affine information does not match")

    if len(trk.streamlines) == 0:
        return False, False
    nodes = extract_nodes(
        trk_file=tract_file,
        nodes=nodes
    )
    mask = get_roi_sections_from_nodes(
        trk_file=tract_file,
        point_array=nodes
    )
    eng_vals = eng_img.get_fdata()
    checkpoints = []
    for value in np.unique(mask):
        coords = np.where(mask == value, 1, 0)
        relevant_values = coords*eng_vals
        checkpoints.append(np.mean(relevant_values))
    unique_vals, inverse = np.unique(mask, return_inverse=True)
    count = np.bincount(inverse)
    count = np.bincount(inverse)
    checkpoints_mean = (
        np.bincount(inverse, weights=eng_vals.ravel()) / count
    )
    checkpoints_mean_squares = (
        (np.bincount(inverse, weights=eng_vals.ravel()**2) / count)
    )
    checkpoints_var = checkpoints_mean_squares - checkpoints_mean**2
    std = np.sqrt(checkpoints_var)

    return (checkpoints_mean, std)

def analyse_all_tracts(
        tract_folder: str,
        output_folder: str,
        engagement_file: str,
        nodes: int, 
        subj: str,
        session: str

):
    all_patient_data = []
    for tract in os.listdir(tract_folder):
        file_extension = tract.split(".")[-1]
        if file_extension != "trk":
            continue
        print(f"Analysing Tract: {tract}")
        patient_tract_data = {"subject": subj, "session": session}
        split_name = tract.split("_")
        tract_name = ""
        for name_part in split_name[3:]:
            tract_name += name_part + "_"
        tract_name = tract_name[:-5]
        patient_tract_data["tract"] = tract_name
        tract_path = join(
            tract_folder,
            tract
        )
        means, stds = tract_engagement(
            tract_file=tract_path,
            engagement_file=engagement_file,
            nodes=nodes
        )
        if means is False or stds is False: 
            print(f"Failure on tract {tract}")
            for i in range(nodes):
                patient_tract_data[f"mu_{i+1}"] = None
                patient_tract_data[f"sigma_{i+1}"] = None
        else:
            for i in range(len(means)):
                patient_tract_data[f"mu_{i+1}"] = means[i]
                patient_tract_data[f"sigma_{i+1}"] = stds[i]
        all_patient_data.append(patient_tract_data)
    patient_frame = pd.DataFrame(
        data=all_patient_data
    )
    csv_name = "TAU-" + subj+ "_" + session +".csv"
    csv_path = join(
        output_folder,
        csv_name
    )
    patient_frame.to_csv(
        csv_path
    )

def correct_atlases(
    mni_path: str,
    atlas_folder: str,
    outputs_folder: str,
    save_prefix: str = "edited_"
):
    """
    Correct `.trk` atlas tractograms by aligning them to an MNI reference image.

    This function iterates over all `.trk` files in a specified folder and
    applies a spatial translation so that their streamline coordinates align
    with the voxel-space origin of a provided MNI reference image. The
    translation offset is derived from the affine matrix of the MNI image.

    For each `.trk` file:
        1. The tractogram is loaded.
        2. Streamlines are converted to voxel space and corner convention.
        3. A translation offset computed from the MNI affine is applied.
        4. A corrected tractogram is saved to the output directory with a
           configurable filename prefix.

    Parameters
    ----------
    mni_path : str
        Path to the reference MNI NIfTI image. Its affine matrix is used
        to compute the spatial translation offset.
    atlas_folder : str
        Path to a directory containing input tractogram files (.trk).
        Only files with the `.trk` extension are processed.
    outputs_folder : str
        Path to the directory where corrected tractograms will be saved.
        The directory is created if it does not exist.
    save_prefix : str, optional
        Prefix added to each corrected tractogram filename before saving.
        Default is "edited_".

    Notes
    -----
    - The correction consists solely of a translation based on the affine
      offset; no rotation or scaling is performed.
    - Streamlines are converted to voxel space (`to_vox()`) and corner
      convention (`to_corner()`) before applying the offset.
    - The output tractograms are saved in voxel space (`Space.VOX`).
    - Files in `atlas_folder` that do not have a `.trk` extension are ignored.
    """
    os.makedirs(
        outputs_folder, 
        exist_ok=True
    )
    mni_image = nib.load(mni_path)
    mni_affine = mni_image.affine
    offsets = -1*mni_affine[0:3, 3]
    offsets -= 0.5 ### @todo Check whether this is valid Maybe add it to github.
    for atlas_file in tqdm(os.listdir(atlas_folder), "Correcting atlases"):
        extension = atlas_file.split(".")[-1]
        if extension!="trk":
            continue
        save_name = os.path.join(
            outputs_folder,
            save_prefix+ atlas_file
        )
        atlas_path = os.path.join(
            atlas_folder,
            atlas_file
        )
        trk = load_tractogram(
            filename=atlas_path, 
            reference="same",
            bbox_valid_check=False
        )
        trk.to_vox()
        trk.to_corner()

        data = trk.streamlines.get_data()
        mins = data.min(axis=0)
        maxs = data.max(axis=0)
        print(trk.space)
        print("Streamline bounds:")
        print("X:", mins[0], "→", maxs[0])
        print("Y:", mins[1], "→", maxs[1])
        print("Z:", mins[2], "→", maxs[2])
        print(mni_image.get_fdata().shape)
        streamlines = trk.streamlines
        data = streamlines._data
        data += offsets 
        mins = data.min(axis=0)
        maxs = data.max(axis=0)
        
        print("Streamline bounds:")
        print("X:", mins[0], "→", maxs[0])
        print("Y:", mins[1], "→", maxs[1])
        print("Z:", mins[2], "→", maxs[2])
        print(mni_image.get_fdata().shape)

        new_trk = StatefulTractogram(
            streamlines=streamlines,
            reference=MNI_PATH,
            space=Space.VOX
        )
        save_tractogram(
            sft=new_trk,
            filename=save_name,
            bbox_valid_check=True
        )


def correct_atlases_V2(
    mni_path: str,
    atlas_folder: str,
    outputs_folder: str,
    save_prefix: str = "edited_"
):
    """
    Correct `.trk` atlas tractograms by aligning them to an MNI reference image.

    This function iterates over all `.trk` files in a specified folder and
    applies a spatial translation so that their streamline coordinates align
    with the voxel-space origin of a provided MNI reference image. The
    translation offset is derived from the affine matrix of the MNI image.

    For each `.trk` file:
        1. The tractogram is loaded.
        2. Streamlines are converted to voxel space and corner convention.
        3. A translation offset computed from the MNI affine is applied.
        4. A corrected tractogram is saved to the output directory with a
           configurable filename prefix.

    Parameters
    ----------
    mni_path : str
        Path to the reference MNI NIfTI image. Its affine matrix is used
        to compute the spatial translation offset.
    atlas_folder : str
        Path to a directory containing input tractogram files (.trk).
        Only files with the `.trk` extension are processed.
    outputs_folder : str
        Path to the directory where corrected tractograms will be saved.
        The directory is created if it does not exist.
    save_prefix : str, optional
        Prefix added to each corrected tractogram filename before saving.
        Default is "edited_".

    Notes
    -----
    - The correction consists solely of a translation based on the affine
      offset; no rotation or scaling is performed.
    - Streamlines are converted to voxel space (`to_vox()`) and corner
      convention (`to_corner()`) before applying the offset.
    - The output tractograms are saved in voxel space (`Space.VOX`).
    - Files in `atlas_folder` that do not have a `.trk` extension are ignored.
    """
    os.makedirs(
        outputs_folder, 
        exist_ok=True
    )
    mni_image = nib.load(mni_path)
    mni_affine = mni_image.affine
    offsets = -1*mni_affine[0:3, 3]
    for atlas_file in tqdm(os.listdir(atlas_folder), "Correcting atlases"):
        extension = atlas_file.split(".")[-1]
        if extension!="trk":
            continue
        save_name = os.path.join(
            outputs_folder,
            save_prefix+ atlas_file
        )
        atlas_path = os.path.join(
            atlas_folder,
            atlas_file
        )
        trk = load_tractogram(
            filename=atlas_path, 
            reference="same",
            bbox_valid_check=False
        )

        sls = trk.streamlines
        new_trk = StatefulTractogram(
            streamlines=sls, 
            reference=mni_path,
            space=Space.RASMM
        )
        save_tractogram(
            sft=new_trk,
            filename=save_name,
            bbox_valid_check=True
        )

def atlas_space_conversion(
        current_space: str,
        target_space: str,
        atlas_folder: str,
        save_destination: str,
        prefix: str = "mni_"
):  
    """
    Converts all trk files in a given folder to a specific target space. 
    The transformation is based only on the affine information. 
    
    :param current_space: Path to a niftii file in space that the 
    tractograms are in
    :type current_space: str
    :param target_space: Path to a niftii image file which is in 
    the target space
    :type target_space: str
    :param atlas_folder: The folder containing trk bundles. 
    :type atlas_folder: str
    :param save_destination: The folder that the modified atlases 
    should be saved to
    :type save_destination: str
    :param prefix: The prefix to attach to the atlas once it has 
    been converted
    """
    os.makedirs(
        save_destination, 
        exist_ok=True
    )
    for atlas_file in tqdm(os.listdir(atlas_folder),"Space conversion"):
        extension = atlas_file.split(".")[-1]
        if extension != "trk":
            continue
        save_name = os.path.join(
            save_destination,
            prefix + atlas_file
        )
        atlas_path = os.path.join(
            atlas_folder,
            atlas_file
        )
        """aligned_atlas = streamline_registration(
            moving_file=current_space,
            static_file=target_space,
            trk_file=atlas_path,
            save=False
        )"""

        trk = load_tractogram(
            filename=atlas_path,
            reference="same",
        )

        tform = find_transform(
            moving_file=current_space,
            static_file=target_space,
            only_affine=True
        )
        new_streamlines =transform_streamlines(
            streamlines=trk.streamlines,
            mat=tform.affine
        )

        data =new_streamlines.get_data()

        mins = data.min(axis=0)
        maxs = data.max(axis=0)

        print("Streamline bounds:")
        print("X:", mins[0], "→", maxs[0])
        print("Y:", mins[1], "→", maxs[1])
        print("Z:", mins[2], "→", maxs[2])

        new_trk = StatefulTractogram(
            streamlines=new_streamlines,
            reference=target_space,
            space=Space.RASMM
        )
        save_tractogram(
            sft = new_trk,
            filename=save_name
        )

def create_tcks(atlas_folder):
    """
    Iterates through a folder containing trk files and creates corresponding
    tck file in the same folder. 
    
    :param atlas_folder: Filepath to the folder containing the trk files.
    """
    for atlas_file in os.listdir(atlas_folder):
        extension = atlas_file.split(".")[-1]
        if extension != "trk":
            continue
        atlas_path = os.path.join(
            atlas_folder,
            atlas_file
        )
        trk2tck(
            input_file=atlas_path,
            bounding=False
        )

def patient_registration(
        atlas_folder,
        original_space,
        target_file,
        output_folder,
        subj,
        verbose: bool=False
):
    os.makedirs(
        output_folder, 
        exist_ok=True
    )
    """if os.path.exists(TEST):
        transform_mat = np.load(TEST)
    else:
        tform = find_transform(
                moving_file=original_space,
                static_file=target_file,
                diffeomorph=False
        )
        transform_mat = tform.affine
        np.save(TEST, tform.affine)"""
    tform = find_transform(
                moving_file=original_space,
                static_file=target_file,
                diffeomorph=False
    )
    
    apply_transform(
        moving_file=original_space,
        static_file=target_file,
        mapping=tform,
        output_path="/Users/sam/Desktop/mni_patient_space.nii.gz"
    )

    for atlas_file in tqdm(os.listdir(atlas_folder)):
        extension = atlas_file.split(".")[-1]
        if extension != "trk":
            continue
        
        fp = join(
            atlas_folder,
            atlas_file
        )
        trk = load_tractogram(
            filename=fp,
            reference="same"
        )
        if verbose:
            print(f"Processing {atlas_file}")
            print("Space is: ", trk.space)
            """ trk.to_vox()
            trk.to_corner() """
            print("Space 2 is: ", trk.space)
            data =trk.streamlines.get_data()

            mins = data.min(axis=0)
            maxs = data.max(axis=0)

            print("Streamline bounds (pre transform):")
            print("X:", mins[0], "→", maxs[0])
            print("Y:", mins[1], "→", maxs[1])
            print("Z:", mins[2], "→", maxs[2])

        img = nib.load(target_file)
        if verbose:
            print("IMG Dimensions\n")
            print(img.get_fdata().shape)
            print("trk\n",trk.affine)
            print("img\n", img.affine)

        new_sl = transform_streamlines(
            trk.streamlines,
            mat = np.linalg.inv(tform.affine) ## When time, consider wny inverting worked
        )
        
        new_trk = StatefulTractogram(
            streamlines=new_sl,
            reference=target_file,
            space = Space.RASMM
        )
        if verbose:
            print("Space 3 is: ", new_trk.space)
            data =new_trk.streamlines.get_data()

            mins = data.min(axis=0)
            maxs = data.max(axis=0)

            print("Streamline bounds (post transform, rasm):")
            print("X:", mins[0], "→", maxs[0])
            print("Y:", mins[1], "→", maxs[1])
            print("Z:", mins[2], "→", maxs[2])

        if verbose:
            print("Space 4 is: ", new_trk.space)
            data =new_trk.streamlines.get_data()

            mins = data.min(axis=0)
            maxs = data.max(axis=0)

            print("Streamline bounds (post transform):")
            print("X:", mins[0], "→", maxs[0])
            print("Y:", mins[1], "→", maxs[1])
            print("Z:", mins[2], "→", maxs[2])

            img = nib.load(target_file)
            print("trk\n",new_trk.affine)
            print("img\n", img.affine)
        filename = join(
            output_folder,
            subj + "_" + atlas_file
        )
        # This line is very suspect
        #new_trk.remove_invalid_streamlines()

        save_tractogram(
            sft=new_trk,
            filename=filename,
            bbox_valid_check=True
        )
        """tform = find_transform(
            moving_file=original_space,
            static_file=target_file,
            diffeomorph=False
        )
        print("transform",tform)
        new_sls = transform_streamlines(
            trk.streamlines,
            mat = tform.affine
        )
        new_trk = StatefulTractogram(
            streamlines=new_sls,
            reference=target_file,
            space=Space.RASMM
        )
        print("Space = ", new_trk.space)
        ref_img = nib.load(target_file)
        print("img affine",ref_img.affine)
        data =new_trk.streamlines.get_data()

        mins = data.min(axis=0)
        maxs = data.max(axis=0)

        print("Streamline bounds:")
        print("X:", mins[0], "→", maxs[0])
        print("Y:", mins[1], "→", maxs[1])
        print("Z:", mins[2], "→", maxs[2])

        new_trk.to_vox()
        new_trk.to_corner()
        data =new_trk.streamlines.get_data()

        mins = data.min(axis=0)
        maxs = data.max(axis=0)

        print("Streamline bounds:")
        print("X:", mins[0], "→", maxs[0])
        print("Y:", mins[1], "→", maxs[1])
        print("Z:", mins[2], "→", maxs[2])
        filename = join(
            output_folder,
            subj + "_" + atlas_file
        )
        save_tractogram(
            sft=new_trk,
            filename=filename,
            bbox_valid_check=True
        )"""


def analyse_dataset(dataset_fp):
    df = pd.read_csv(dataset_fp)
    unique_tracts = df["tract"].unique()

    for unique_tract in unique_tracts:
        subset = df[df["tract"] == unique_tract]
        patients = subset["subject"]
        print(patients.shape)
        tract = subset["tract"]
        subset = subset.filter(regex = "mu")
        subset = subset.fillna(0)

        model = NMF(
            n_components='auto',
            init='random',
            random_state=0
        )
        W = model.fit_transform(subset.to_numpy())

        headers = model.get_feature_names_out()
        tract_df = pd.DataFrame(
            data=W, 
            columns = headers
        )
        tract_df["subject"] = patients
        tract_df["tract"] = tract
       

if __name__ == "__main__":
    """correct_atlases(
        mni_path=MNI_PATH,
        atlas_folder=ATLAS_FOLDER,
        outputs_folder=NEW_ATLAS_FP
    )
    create_tcks(NEW_ATLAS_FP)"""
    """
    atlas_space_conversion(
        MNI_PATH,
        MNI_PATH_MINE,
        NEW_ATLAS_FP,
        OUR_MNI_BUNDLES
    )"""

    patient_registration(
        atlas_folder=OUR_MNI_BUNDLES,
        original_space=MNI_PATH_MINE,
        target_file=T1_ANAT_FILE,
        output_folder=PATIENT_FOLDER,
        subj="TAU001",
        verbose=True
    )
    create_tcks(PATIENT_FOLDER)
    """
    analyse_all_tracts(
        tract_folder=WITH_INVERTED,
        output_folder=ENG_STORAGE,
        engagement_file=PATIENT_ENG,
        nodes = 30,
        subj="001",
        session="ses-2"
    )"""
   
    """analyse_dataset(
        ENG_STORAGE + "/TAU-001_ses-2.csv"
    )"""

