import numpy as np
from iCliff.iCliff import calculate_comp_iCliff, pair_prop_diff, mol_set_tanimoto, ts_sali_matrix
import pandas as pd
import glob

fp_type = 'ECFP4'
for file in glob.glob('data/props/*.npy'):
    # Get the name
    name = file.split('/')[-1].split('.')[0]

    # Read the fingerprints data
    fps = np.load(f'data/fps/{fp_type}/{name}.npy', mmap_mode='r')

    # Read the smiles data
    smiles = pd.read_csv(f'data/smiles/{name}.csv')
    props = smiles['exp_mean [nM]'].values
    smiles = smiles['smiles'].values
    props = props/10**9
    props = -np.log10(props)

    # Read the ts_sali_matrix data
    ts_sali = np.load(f'data/ts_sali_matrix/{fp_type}/{name}.npy', mmap_mode='r')

    # Calculate the comp_iCliff values
    comp_iCliff = calculate_comp_iCliff(fps, props)

    # Sort the iCliff values, get the 1% with lowest comp iCliff values
    idx = np.argsort(comp_iCliff)
    idx = idx[int(len(idx)*0.01):]

    smiles_1 = []
    smiles_2 = []
    props_1 = []
    props_2 = []
    sim = []
    ts_salis = []
    # Save the instances where the TS_SALI values are the in the top 10 for each compound in the rows
    for id in idx:
        # Get the top 10% of the ts_sali values for each compound
        top = np.percentile(ts_sali[id], 90)

        # Get the indices of the top 10% values
        top_indices = np.argwhere(ts_sali[id] > top)
        top_indices = top_indices[:, 0]

        # Get the ts_sali values of the problematic compound and their 10% according to TSSALI
        ts_sali_prob = ts_sali[id][top_indices]

        # Get the smiles of the problematic compound and their 10% according to TSSALI
        smiles_prob = smiles[id]
        smiles_top = smiles[top_indices]

        # Replicate the problematic compound smiles as many times as the top 10% according to TSSALI
        smiles_prob = np.array([smiles_prob]*len(smiles_top))

        # Get the properties of the problematic compound and their 10% according to TSSALI
        props_prob = props[id]
        props_top = props[top_indices]

        # Get the fps of the problematic compound and their 10% according to TSSALI
        fps_prob = fps[id]
        fps_top = fps[top_indices]

        # Get the similarity of the problematic compound and their 10% according to TSSALI
        sim_prob = mol_set_tanimoto(fps_prob, fps_top)

        # Append the values to the lists
        smiles_1.append(smiles_prob)
        smiles_2.append(smiles_top)
        props_1.append(props_prob)
        props_2.append(props_top)
        sim.append(sim_prob)
        ts_salis.append(ts_sali_prob)

    # Save the values to a dataframe
    df = pd.DataFrame({
        'smiles_1': smiles_1,
        'smiles_2': smiles_2,
        'props_1': props_1,
        'props_2': props_2,
        'sim': sim,
        'ts_sali': ts_salis
    })

    # Save the dataframe to a csv file
    df.to_csv(f'results/pair_identification/{name}_iCliff.csv', index=False)

    # Print finished
    print(f'Finished {name} iCliff analysis')

        
