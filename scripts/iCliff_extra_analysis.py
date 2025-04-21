import numpy as np
from iCliff.iCliff import calculate_comp_iCliff, squared_prop_diff_matrix, tanimoto_similarity_matrix, ts_sali_matrix, tanimoto_similarity
import pandas as pd
import glob

fp_type = 'ECFP4'
for file in glob.glob('data/props/*.npy'):
    # Get the name
    name = file.split('/')[-1].split('.')[0]

    # Read the props data
    props = np.load(file, mmap_mode='r')

    # Read the fingerprints data
    fps = np.load(f'data/fps/{fp_type}/{name}.npy', mmap_mode='r')

    # Read the smiles data
    smiles = pd.read_csv(f'data/smiles/{name}.csv')
    smiles = smiles['smiles'].values

    # Calculate the comp_iCliff values
    comp_iCliff = calculate_comp_iCliff(fps, props)

    # Sort the iCliff values, get the 10% with lowest comp iCliff values
    idx = np.argsort(comp_iCliff)
    idx = idx[int(len(idx)*0.1):]

    # Get the properties and fingerprints of the 10% with lowest comp iCliff values
    props_ = props[idx]
    fps_ = fps[idx]
    smiles_ = smiles[idx]

    # Calculate the squared property difference matrix
    prop_diff_matrix = squared_prop_diff_matrix(props_)

    # Calculate the Tanimoto similarity matrix
    sim_matrix = tanimoto_similarity_matrix(fps_)

    # Calculate the ts_sali matrix
    ts_sali = ts_sali_matrix(prop_diff_matrix, sim_matrix, term=3)

    # Find the positions where you find the 10% highest values of the ts_sali matrix
    top = np.percentile(ts_sali, 90)
    top_positions = np.argwhere(ts_sali > top)
    
    entries = []
    # Create the data entries
    for i, j in top_positions:
        # Get the smiles of the two compounds
        smiles_1 = smiles_[i]
        smiles_2 = smiles_[j]

        # Get the properties of the two compounds
        props_1 = props_[i]
        props_2 = props_[j]

        # Get the similarity of the two compounds
        sim = tanimoto_similarity(fps_[i], fps_[j])

        # Get the ts_sali values
        iCliff_pair = ts_sali[i, j]

        # Append the data
        entries.append([idx[i], idx[j], smiles_1, smiles_2, props_1, props_2, sim, iCliff_pair])

    # Save the results
    df = pd.DataFrame(entries, columns = ['idx_1', 'idx_2', 'smiles_1', 'smiles_2', 'props_1', 'props_2', 'similarity', 'iCliff'])
    df.to_csv(f'results/pair_identification/{name}_iCliff.csv', index=False)

    print(f'Finished {name}')
