import numpy as np
from scipy.stats import kendalltau
from iCliff.iCliff import ts_sali_matrix, calculate_iCliff
import glob
import pandas as pd

entries = []
for file in glob.glob('/blue/rmirandaquintana/klopezperez/ecliffs/fps/CHEMBL*.pkl'):
    # Get the name
    name = file.split('/')[-1].split('.')[0]

    # Read the data
    data = pd.read_pickle(file)

    # Read the properties and fps
    props = data['prop']

    # Read if its an AC or not
    ac = np.array(data['cliff'])
    n_ac = np.sum(ac) 

    # Convert and normalize the properties
    props = np.array(props)/10**9
    props = -np.log(props)

    # Normalize the properties
    props = (props - np.min(props))/(np.max(props) - np.min(props))


    # Read prop_diffs
    props_diffs = np.load(f'/blue/rmirandaquintana/klopezperez/ecliffs/pair_matrices/prop_diffs/{name}.npy')

    for fp in ['RDKIT', 'ECFP', 'MACCS']:
        # Read fps
        fps = np.array(data[fp])

        # Read the similarity matrix
        sim = np.load(f'/blue/rmirandaquintana/klopezperez/ecliffs/pair_matrices/JT/{fp}/{name}_{fp}_JT.npy')

        # Calculate the Taylor sum SALI matrix
        ts_sali = ts_sali_matrix(props_diffs, sim, term = 3)

        # Get the average ts_sali
        avg_ts_sali = np.mean(ts_sali)

        # Get the iCliff global
        iCliff_global = calculate_iCliff(fps, props)

        entried.append([name, fp, iCliff, avg_ts_sali])

# Save the results
df = pd.DataFrame(entries, columns = ['name', 'fp', 'iCliff', 'ts_sali_avg'])
df.to_csv('iCliff_ts_sali.csv', index = False)
