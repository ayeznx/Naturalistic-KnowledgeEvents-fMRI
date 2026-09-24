##### Imports

# Basic Imports
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import os

# Stats
import numpy as np
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import train_test_split
from nilearn.image import threshold_img
from brainiak.image import mask_image
from data_prep_helpers_new import make_vol, make_boolean_mask


##### Functions

def rand_sign_flip(voxels):
    """
        Randomly flips signs for some indexs of an array of voxels
        Generates an random array of [1,-1] with the same length as voxels
        Element-wise multiplication of the two arrays inverts signs for random voxels
    """
    flip_indexes = [np.random.choice([-1,1]) for i in range(len(voxels))]
    flipped_voxels = np.multiply(voxels, flip_indexes)
    return flipped_voxels

def generate_null(data, n_permutations):
    """
        Generates a null distribution via a 1-sampled sign-flip permutation test
        Returns a null distribution of dim n_permutations x n_voxels
    """

    null_dist = []

    for i in range(n_permutations):

        # Randomly flip signs for each subject's data
        flipped_data = []
        for subject in range(len(data)):
            flipped_data.append(rand_sign_flip(data[subject]))

        # Calculate mean from sign flipped subject data
        flipped_mean = np.mean(flipped_data, axis=0)

        # Store as a row in null_dist
        null_dist.append(flipped_mean)
    return null_dist

def generate_null_comparison(group1, group2, n_permutations):
    """
        Generates a null distribution of inter-group differences via signed permutation test.
        Returns a null distribution of dim n_voxels x n_permutations
    """
    null_dist = []
    for i in range(n_permutations):
        flipped_group1= []
        for subject in range(len(group1)):
            flipped_group1.append(rand_sign_flip(group1[subject]))

        flipped_group2= []
        for subject in range(len(group2)):
            flipped_group2.append(rand_sign_flip(group2[subject]))

        # Calculate difference in means
        null_dist.append(np.subtract(np.mean(flipped_group1, axis=0), np.mean(flipped_group2, axis=0)))
    return null_dist

def generate_null_permute_groups(group1, group2, n_permutations):
    """
        Generates a null distribution of inter-group differences via permutation test.
        Returns a null distribution of dim n_voxels x n_permutations
    """
    all_subjects = np.vstack((group1, group2))
    null_dist = []
    for i in range(n_permutations):
        # Split data
        sample_group1, sample_group2 = train_test_split(all_subjects, train_size=len(group1), test_size=len(group2))

        # Calculate difference in means
        null_dist.append(np.subtract(np.mean(sample_group1, axis=0), np.mean(sample_group2, axis=0)))
    return np.swapaxes(null_dist, 0, 1)

def generate_null_difference_clustered(group1, group2, n_permutations, cluster_size, mask):
    """
        Generates a null distribution of inter-group differences via signed permutation test,
        then thresholds based on cluster size, with a minimum correlation of 0.
        Returns a null distribution of dim n_voxels x n_permutations
    """
    null_dist = []
    for i in range(n_permutations):
        flipped_group1= []
        for subject in range(len(group1)):
            flipped_group1.append(rand_sign_flip(group1[subject]))

        flipped_group2= []
        for subject in range(len(group2)):
            flipped_group2.append(rand_sign_flip(group2[subject]))

         # Calculate difference in means
        diff = np.subtract(np.mean(flipped_group1, axis=0), np.mean(flipped_group2, axis=0))

        # Cluster correct
        clustered_null_vol = threshold_img(img=make_vol(diff, mask),
                                       threshold=0.0, cluster_threshold=cluster_size, two_sided=False)
        clustered_null_voxels = mask_image(clustered_null_vol, make_boolean_mask(mask))

        null_dist.append(clustered_null_voxels)
    return null_dist

def generate_null_difference_permute_groups_clustered(group1, group2, n_permutations, cluster_size, mask):
    """
        Generates a null distribution of inter-group differences via group label  permutation test,
        then thresholds based on cluster size, with a minimum correlation of 0.
        Returns a null distribution of dim n_voxels x n_permutations
    """
    all_subjects = np.vstack((group1, group2))
    null_dist = []
    for i in range(n_permutations):
        # Split data
        sample_group1, sample_group2 = train_test_split(all_subjects, train_size=len(group1), test_size=len(group2))

        # Calculate difference in means
        diff = np.subtract(np.mean(sample_group1, axis=0), np.mean(sample_group2, axis=0))

        # Cluster correct
        clustered_null_vol = threshold_img(img=make_vol(diff, mask),
                                       threshold=0.0, cluster_threshold=cluster_size, two_sided=False)
        clustered_null_voxels = mask_image(clustered_null_vol, make_boolean_mask(mask))

        null_dist.append(clustered_null_voxels)
    return null_dist

def get_p_values(observed, null_dist):
    """
        Generates one-tailed p-value for each score relative to the null distribution
        Assumes null_dist takes the shape [n_voxels, n_iterations]
    """
    p_vals = []
    num_iterations = np.shape(null_dist)[1]
    print('Num iterations:', num_iterations)
    print('Observed:', np.shape(observed))

    # Take mean if not already a single avg set of scores
    if observed.ndim > 1:
        print('Averaging scores!')
        observed = np.mean(observed, axis=0)

    for i in range(len(observed)):

        # Extract relevant distribution from overall data
        observed_score = observed[i]
        null_scores = null_dist[i]

        # Get number of values greater than observed_score
        more_extreme_count = len(np.where(null_scores >= observed_score)[0])

        # Calculate p-value
        p_val = (more_extreme_count + 1) / (num_iterations + 1)

        # Store p-value
        p_vals.append(p_val)

    return p_vals

def get_fdr_controlled(p, threshold, observed):
    """
        Given a set of p values for each voxel and a threshold, applies FDR correction
        Returns number of significant voxels and array with only significant voxels retained
    """
    # Make changes to a copy of the original array
    if observed.ndim > 1:
        thresholded_voxels = np.mean(observed, axis=0)
    else:
        thresholded_voxels = observed

    # Generate fdr corrected p values (i,e q values)
    q = multipletests(p, method='fdr_bh')[1]

    # Mask insignificant voxels out with NaNs
    thresholded_voxels[q >= threshold] = np.nan

    # Calculate the number of significant voxels
    num_significant_voxels = np.sum(q < threshold)

    return thresholded_voxels, num_significant_voxels

def concatenate_null_dists(directory):
    """
        Concatenates null distributions from multiple .npy files into a single array.
        Assumes all files are stored in directory, will ignore non-.npy files
        Returns an array of dimensions num_dists x num_voxels
    """
    all_files = os.listdir(directory)
    distributions = []

    # Ensure only .npy files are stored (remove .DS_Store, etc)
    for filename in all_files:
        if filename[-4:] in ('.npy'):
            dist_name = os.path.join(directory, filename)
            distributions.append(np.load(dist_name))
        else:
            print('discarding file: ', filename)

    # Perform concatenation
    concatenated_dist = []
    for i in range(len(distributions)):
        for dist in distributions[i]:
            concatenated_dist.append(dist)

    # Return in a useful format for subsequent functions
    return np.swapaxes(concatenated_dist, 0, 1)


def perform_fdr_correction(scores, null_dist_path, threshold):
    """
    Given a set of scores and a directory of null distributions, concatenate the distributions
    and perform FDR correction at the specified threshold.
    """
    # Concatenate into a single null distribution
    null_dists = concatenate_null_dists(null_dist_path)

    # Generate FDR controlled values
    p = get_p_values(scores, null_dists)
    voxels, num_significant = get_fdr_controlled(p, threshold=threshold, observed=scores)

    return voxels, num_significant

def fdr_correct_comparison(group1, group2, difference, threshold, permutations):
    """
    Given performance for 2 groups, generate null distribution for the difference between them,
    and perform FDR correction at the specified threshold.
    """
    # Concatenate into a single null distribution
    null_dists = generate_null_comparison(group1, group2, n_permutations=permutations)
    p = get_p_values(difference, null_dists)
    voxels, num_significant = get_fdr_controlled(p, threshold=threshold, observed=difference)

    return voxels, num_significant


