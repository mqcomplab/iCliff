import numpy as np
from iCliff.iCliff import calculate_iCliff, ts_sali_max, sali_analysis, ts_sali_matrix, jaccard
import pandas as pd
import glob

fp_type = 'ECFP'
activity_cliffs = []
for file in glob.glob('/blue/rmirandaquintana/klopezperez/ecliffs/fps/CHEMBL*'):
    # Get the name
    name = file.split('/')[-1].split('.')[0]

    # Read the data
    data = pd.read_pickle(f'/blue/rmirandaquintana/klopezperez/ecliffs/fps/{file.split(".")[0]}.pkl')

    # Read the properties and fps
    props = data['prop']
    fps = data[fp_type]

    del data

    # Convert and normalize the properties
    props = np.array(props)/10**9
    props = -np.log(props)

    # Normalize the properties
    props = (props - np.min(props))/(np.max(props) - np.min(props))

    # Calculate the iCliff values
    iCliff = calculate_iCliff(props, fps)

    # Find the indexes of the 10 lowest iCliff values
    idx = np.argsort(iCliff)[:10]

    # Calculate the ts_sali
    prop_matrix = np.load(f'/blue/rmirandaquintana/klopezperez/ecliffs/pair_matrices/prop_diffs/CHEMBL*', mmap_mode='r')
    sim_matrix = np.load(f'/blue/rmirandaquintana/klopezperez/ecliffs/pair_matrices/JT/{fp_type}/{file.split(".")[0]}_{fp_type}_JT.npy', mmap_mode='r')

    ts_sali = ts_sali_matrix(props, fps, term=3)

    del prop_matrix, sim_matrix

    # Keep the ts_sali rows that are in the idx
    ts_sali = ts_sali[idx]

    # Find the highest pair activity cliffs
    for id, row in zip(idx, ts_sali):
        # Find the highest ts_sali value for that row
        a = np.argmax(row)
        activity_cliffs.append([name, id, a, props[id], props[a], prop_matrix[id, a], sim_matrix[id, a]])
    
# Save the preliminary results
df = pd.DataFrame(activity_cliffs, columns = ['name', 'ac_1', 'ac_2', 'prop_1', 'prop_2', 'prop_diff_squared', 'similarity'])
df.to_csv('activity_cliffs.csv', index = False)

smiles_one = []
smiles_two = []
# Read the csv with the smiles and append those columns to the dataframe
for database in np.unique(df['name']):
    df_ = df[df['name'] == database]

    drop_name = database.split('_f')[0]
    data = pd.read_csv(f'/blue/rmirandaquintana/klopezperez/ecliffs/csv_smiles/{database}.csv')

    # Get the smiles
    smiles = data['smiles']

    # Get the smiles of the activity cliffs
    smiles_1 = smiles[df_['ac_1']]
    smiles_2 = smiles[df_['ac_2']]

    smiles_one.extend(smiles_1)
    smiles_two.extend(smiles_2)

    del data

df['smiles_1'] = smiles_one
df['smiles_2'] = smiles_two

# Save the final results
df.to_csv('activity_cliffs.csv', index = False)