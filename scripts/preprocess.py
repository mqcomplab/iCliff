import glob as glob
import numpy as np
import pandas as pd
from iCliff.utils import binary_fps_numpy, rdkit_pairwise_matrix

# Script to preprocess the properties of the molecules in the sets
for file in glob.glob('data/*.csv'):
    # Get the name of the database
    name = file.split('/')[-1].split('.')[0]

    # Read the smiles data
    data = pd.read_csv(file)

    ##################################
    ####### Property Processing ######
    ##################################

    # Get the values of the properties
    props = data.loc[:, 'exp_mean [nM]']

    # Convert from nM to M
    props = props * 1e-9

    # Calculate the pKi or the pEC50
    props = -np.log10(props)

    # Save the properties
    np.save('data/props/' + name + '.npy', props)

    ##################################
    ####### Fingerprint Processing ###
    ##################################
    # Get the smiles
    smiles = data.loc[:, 'smiles']

    # Compute the fingerprints and calculate the pairwise similarity matrix
    for fp_type in ["MACCS", "ECFP4", "RDKIT"]:
        fps = binary_fps_numpy(smiles, fp_type, n_bits=1024)
        np.save(f'data/fps/{fp_type}/{name}.npy', fps)

        # Convert the fingerprints to RDKit format
        pairwise_matrix = rdkit_pairwise_matrix(fps)

        # Save the pairwise similarity matrix
        np.save(f'data/sim_matrix/{fp_type}/{name}.npy', pairwise_matrix)
