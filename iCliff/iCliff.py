import numpy as np
import pandas as pd
#from comp import *
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
import seaborn as sns

## Utility functions

def jaccard(x, y):
    """Calculate the Jaccard similarity between two sets."""
    return len(np.intersect1d(x,y))/len(np.union1d(x,y))

def tanimoto_similarity(fp1, fp2):
    """
    Calculate the Tanimoto similarity between two fingerprints.
    
    Parameters:
        fp1 (numpy.ndarray): The first fingerprint.
        fp2 (numpy.ndarray): The second fingerprint.
        
    Returns:
        float: The Tanimoto similarity between the two fingerprints.
    """
    # Compute the dot product
    dot_product = np.dot(fp1, fp2)
    
    # Calculate the Tanimoto similarity
    tanimoto = dot_product / (np.sum(fp1) + np.sum(fp2) - dot_product)
    
    return tanimoto


def jt_isim(c_total, n_objects):
    """iSIM Tanimoto calculation
    
    https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00041b
    
    Parameters
    ----------
    c_total : np.ndarray
              Sum of the elements column-wise
              
    n_objects : int
                Number of elements
                
    Returns
    ----------
    isim : float
           iSIM Jaccard-Tanimoto value
    """
    sum_kq = np.sum(c_total)
    sum_kqsq = np.dot(c_total, c_total)
    a = (sum_kqsq - sum_kq)/2

    return a/(a + n_objects * sum_kq - sum_kqsq)

def calculate_comp_sim(fps):
    """Calculate the complementary similarity for RR, JT, or SM

    Arguments
    ---------
    fps : np.ndarray
        Array of arrays, each sub-array contains the binary object 
        
    Returns
    -------
    comp_sims : nd.array
        1D array with the complementary similarities of all the molecules in the set.
    """
    
    n_objects = len(fps) - 1
    c_total = np.sum(fps, axis = 0)
    comp_matrix = c_total - fps
    a = comp_matrix * (comp_matrix - 1)/2
    
    comp_sims = np.sum(a, axis = 1)/np.sum((a + comp_matrix * (n_objects - comp_matrix)), axis = 1)

    return comp_sims

def tanimoto_similarity_matrix(fps):
    """
    Calculate the Tanimoto similarity matrix for a set of fingerprints.
    
    Parameters:
        fps (numpy.ndarray): The fingerprint matrix.
        
    Returns:
        numpy.ndarray: The Tanimoto similarity matrix.
    """
    # Compute the pairwise dot products
    dot_products = np.dot(fps, fps.T)

    # Get the fingerprint cardinalities
    fps_card = fps.sum(axis=1, keepdims=True)

    # Calculate the Tanimoto similarity
    tanimoto_matrix = dot_products / (fps_card + fps_card.T - dot_products)

    return tanimoto_matrix

def pair_prop_diff(p, props):
    """Calculate the squared differences between a property and all the other property values in the set."""
    return (props - p)**2

def pair_prop_diff_matrix(props):
    """Calculate the pairwise squared differences between properties."""
    return np.square(props[:, None] - props)

def sort_indices(arr, order='asc'):
    """
    Returns the indices that would sort the array.
    
    Parameters:
        arr (numpy.ndarray): The array to sort.
        order (str): Sorting order - 'asc' for ascending, 'desc' for descending.
        
    Returns:
        numpy.ndarray: Sorted indices of the array.
    """
    if order == 'asc':
        # Sort indices in ascending order
        return np.argsort(arr)
    elif order == 'desc':
        # Sort indices in descending order
        return np.argsort(arr)[::-1]
    else:
        raise ValueError("Order must be 'asc' or 'desc'.")

def remove_top_k(arr, k, remove_max=True):
    """
    Removes the top k max or min values from a 1D NumPy array.

    Parameters:
    arr (np.ndarray): 1D NumPy array
    k (int): Number of values to remove
    remove_max (bool): If True, remove top k max values, otherwise remove top k min values

    Returns:
    np.ndarray: Array with top k values removed (max or min)
    """
    # Check if k is valid
    if k <= 0:
        return arr
    if k >= len(arr):
        return np.array([])  # If k is greater than or equal to array length, return an empty array
    
    if remove_max:
        # Find the indices of the top k max values
        top_k_indices = np.argpartition(arr, -k)[-k:]
    else:
        # Find the indices of the top k min values
        top_k_indices = np.argpartition(arr, k)[:k]
    
    # Remove the top k values by masking them
    mask = np.ones(len(arr), dtype=bool)
    mask[top_k_indices] = False
    
    # Return the array without the top k values
    return arr[mask]

# Calculate pairwise similarities between one molecule and a set of molecules
def mol_set_tanimoto(mol, mol_set):
    """Calculate the Tanimoto similarity between a molecule and a set of molecules.
    Returns a vector of similarities between the molecule and each molecule in the set."""
    a = np.dot(mol_set, mol)
    sim_matrix = a / (np.sum(mol_set, axis = 1) + np.sum(mol) - a)
    return sim_matrix

def sali_analysis(ts_sali_matrix, icliff_vector, out_name = 'sali_analysis.csv'):
    """Compares the rankings of molecules in the SALI and iCliff orders."""

    # Sum columnwise the SALI matrix
    sali_vector = np.sum(ts_sali_matrix, axis = 0)

    if len(sali_vector.shape) != 1 or len(icliff_vector.shape) != 1:
        sali_vector = sali_vector.flatten()
        icliff_vector = icliff_vector.flatten()
    
    sali_indices = sort_indices(sali_vector, order ='desc')
    icliff_indices = sort_indices(icliff_vector, order ='asc')
    
    # For a given molecule in the ith position of the SALI order
    # how many positions in the iCliff order this same molecule appears
    diffs = []
    for i in range(len(sali_indices)):
        val = sali_indices[i]
        j = np.where(icliff_indices == val)[0][0]
        diffs.append(abs(i - j))
    
    # Set Jaccard comparisons of slices of the SALI and iCliff orders
    jvs = []
    for i in range(len(sali_indices)):
        jac = jaccard(sali_indices[:i+1], icliff_indices[:i+1])
        jvs.append(jac)
    
    # Create df
    df = pd.DataFrame({'position_diff': diffs, 'jaccard': jvs})
    df.to_csv(out_name, index = False)

    return df

def sali_matrix(prop_matrix, sim_matrix):
    """Calculate the SALI matrix from the property and similarity matrices."""
    sali = prop_matrix/(1 - sim_matrix)
    np.fill_diagonal(sali, 0)
    
    return sali

def ts_sali_matrix(prop_matrix, sim_matrix, term = 2):
    """Calculate the Taylor sum SALI matrix from the property and similarity matrices."""
    if term == 1:
        ts_sali = prop_matrix * (1 + sim_matrix)/2
    elif term == 2:
        ts_sali = prop_matrix * (1 + sim_matrix + sim_matrix**2)/3
    elif term == 3:
        ts_sali = prop_matrix * (1 + sim_matrix + sim_matrix**2 + sim_matrix**3)/4
    else:
        raise ValueError('Unsupported term value. Choose 1, 2 or 3.')    
    return ts_sali

def calculate_iCliff(fps, props):
    """Calculate the iCliff values for a set of fingerprints and properties."""
    N = len(fps)

    # Calculate iSIM Tanimoto
    iSIM = jt_isim(np.sum(fps, axis = 0), N)

    # Calculate the average squared differences between the properties
    prop_diffs = 2 * (np.sum((props**2)/N) - (np.sum(props)/N)**2)

    # Calculate the iCliff values
    iCliff = prop_diffs * (1 + iSIM + iSIM**2 + iSIM**3)/4
    
    return iCliff

def calculate_comp_iCliff(fps, props):
    """Calculate the iCliff values for a set of fingerprints and properties."""
    # Calculate the complementary similarities
    csims = calculate_comp_sim(fps)

    # Get the number of fingerprints
    nfps = len(fps)

    # Check if the properties are normalized
    if np.max(props) > 1 or np.min(props) < 0:
        props = (props - np.min(props))/(np.max(props) - np.min(props))

    # Squared values of the properties
    props_sq = props**2

    # Sum of all properties
    s_props = np.sum(props)

    # Sum of the squares of the properties
    s_props_sq = np.sum(props_sq)

    # Sum of squared errors after removing each molecule
    props_dev = (s_props_sq - props_sq)/(nfps - 1) - ((s_props - props)/(nfps - 1))**2

    # iCliff values for each molecule
    iCliff = props_dev * (1 + csims + csims**2 + csims**3)/4
    
    return iCliff

def ts_sali_max(ts_sali_matrix, frac_max = 0.1):
    # Identify max value in the ts_sali matrix
    max_ts_sali = np.max(ts_sali_matrix)
 
    # Threshold
    s_threshold = max_ts_sali * frac_max

    # Modify SALI matrix to only have 1's for activity cliffs and 0 elsewhere
    ts_sali_matrix[ts_sali_matrix >= s_threshold] = 1
    ts_sali_matrix[ts_sali_matrix < s_threshold] = 0

    return ts_sali_matrix

# Detailed process on how to calculate icliff
#file = 'CHEMBL2047_EC50_fp.pkl'

#obj = pd.read_pickle(file)

#fp_type = 'ECFP'

# Fingerprints
#fps = obj[fp_type]

# Number of molecules
#nfps = len(fps)

# Properties
#props = np.array(obj['prop'])
#props = np.log(props)

# Normalizing
#props = (props - np.min(props))/(np.max(props) - np.min(props))

# Squared values of the properties
#props_sq = props**2

# Sum of all properties
#s_props = np.sum(props)

# Sum of the squares of the properties
#s_props_sq = np.sum(props_sq)

# Sum of squared errors after removing each molecule
#props_dev = (s_props_sq - props_sq)/(nfps - 1) - ((s_props - props)/(nfps - 1))**2

# Complementary similarities
#csims = calculate_comp_sim(fps, n_ary='JT')

# iCliff values for each molecule

#icliff_s = props_dev * (1 + csims + csims**2)/3

# Analysis of the correspondence of the rankings of both SALI types for the global ACs (sum of elements of the SALI matrix)
# These functions generate files with two columns:
# First column: difference in ranking of molecules sorted with SALI and iCliff (the lower the number the better)
# Second column: Set Jaccard similarity of the molecules selected by SALI and iCliff
#ts_sali_total = np.sum(ts_sali, axis=0)
#sali_analysis(ts_sali_total, icliff_s, out_name = 'global_ss') # This seems to do a much better job

# Local SALI analysis
# This is the classical AC study, where only individual values in the matrix that are above a threshold are taken into account
# Given the differences in the product and sum SALI formulas, it makes sense to consider different thresholds for both

# iCliff_s analysis

# Set threshold

# Identify max value in the ts_sali matrix
#max_ts_sali = np.max(ts_sali)

# Up to which % of the max we are going to consider the ACs
#frac_max = 0.1 # For ~0.9-0.6 there are just too few ACs for this to be meaningful. At 0.4 the results become pretty good

# Threshold
#s_threshold = max_ts_sali * frac_max

# Modify SALI matrix to only have 1's for activity cliffs and 0 elsewhere
#ts_sali[ts_sali >= s_threshold] = 1
#ts_sali[ts_sali < s_threshold] = 0

# Calculate number of activity cliffs/molecule
#ac_summary_s = np.sum(ts_sali, axis=0)

# Perform SALI analysis
#sali_analysis(ac_summary_s, icliff_s, out_name = 'local_ss')
