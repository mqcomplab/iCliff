import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

"""
This module contains utility functions for the iChem package regarding fingerprint generation, and 
pairwise similarity calculations using RDKit functions.
"""
def binary_fps_numpy(smiles: list, fp_type: str = 'RDKIT', n_bits: int = 2048):
    """
    This function generates binary fingerprints for the dataset.
    
    Parameters:
    smiles: list of SMILES strings
    fp_type: type of fingerprint to generate ['RDKIT', 'ECFP4', 'ECFP6', or 'MACCS']
    n_bits: number of bits for the fingerprint
    
    Returns:
    fingerprints: numpy array of fingerprints
    """
    # Generate the fingerprints
    if fp_type == 'RDKIT':
       def generate_fp(mol, fp):
            DataStructs.cDataStructs.ConvertToNumpyArray(Chem.RDKFingerprint(mol), fp)
    elif fp_type == 'ECFP4':
        def generate_fp(mol, fp):
            DataStructs.cDataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits), fp)
    elif fp_type == 'ECFP6':
        def generate_fp(mol, fp):
            DataStructs.cDataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=n_bits), fp)
    elif fp_type == 'MACCS':
        def generate_fp(mol, fp):
            DataStructs.cDataStructs.ConvertToNumpyArray(Chem.rdMolDescriptors.GetMACCSKeysFingerprint(mol), fp)
    else:
        print('Invalid fingerprint type: ', fp_type)
        exit(0)

    fingerprints = []
    for smi in smiles:
        # Generate the mol object
        try:
          mol = Chem.MolFromSmiles(smi)
        except:
          print('Invalid SMILES: ', smi)
          exit(0)

        # Generate the fingerprint and append to the list
        fingerprint = np.array([])
        generate_fp(mol, fingerprint)
        fingerprints.append(fingerprint)
    
    fingerprints = np.array(fingerprints)

    return fingerprints

def npy_to_rdkit(fps_np):
    """
    This function converts numpy array fingerprints to RDKit fingerprints.

    Parameters:
    fps_np: numpy array of fingerprints

    Returns:
    fp_rdkit: list of RDKit fingerprints
    """
    fp_len = len(fps_np[0])
    fp_rdkit = []
    for fp in fps_np:
        bitvect = DataStructs.ExplicitBitVect(fp_len)
        bitvect.SetBitsFromList(np.where(fp)[0].tolist())
        fp_rdkit.append(bitvect)
    
    return fp_rdkit


def rdkit_pairwise_sim(fingerprints):
    """
    This function computes the pairwise similarity between all objects in the dataset using Jaccard-Tanimoto similarity.

    Parameters:
    fingerprints: list of fingerprints

    Returns:
    similarity: average similarity between all objects
    """
    if type(fingerprints[0]) == np.ndarray:
        fingerprints = npy_to_rdkit(fingerprints)

    nfps = len(fingerprints)
    similarity = []

    for n in range(nfps - 1):
        sim = DataStructs.BulkTanimotoSimilarity(fingerprints[n], fingerprints[n+1:])
        similarity.extend([s for s in sim])

    return np.mean(similarity)


def rdkit_pairwise_matrix(fingerprints):
    """
    This function computes the pairwise similarity between all objects in the dataset using Jaccard-Tanimoto similarity.

    Parameters:
    fingerprints: list of fingerprints

    Returns:
    similarity: matrix of similarity values
    """
    if type(fingerprints[0]) == np.ndarray:
        fingerprints = npy_to_rdkit(fingerprints)

    n = len(fingerprints)
    matrix = np.zeros((n, n))
    np.fill_diagonal(matrix, 1)  # Set diagonal values to 1

    # Fill the upper triangle directly while computing similarities
    for i in range(n - 1):
        sim = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[i + 1:])
        for j, s in enumerate(sim):
            matrix[i, i + 1 + j] = s  # Map similarities to the correct indices in the upper triangle

    return matrix