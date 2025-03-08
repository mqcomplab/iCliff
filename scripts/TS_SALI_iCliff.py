import numpy as np
from iCliff.iCliff import ts_sali_matrix, calculate_iCliff, squared_prop_diff_matrix
import glob
import pandas as pd

entries = []
for file in glob.glob('data/props/*.npy'):
    # Get the name
    name = file.split('/')[-1].split('.')[0]

    # Load the properties
    props = np.load(file, mmap_mode='r')

    # Calculate the squared diff property matrix
    props_diffs = squared_prop_diff_matrix(props)

    for fp in ['RDKIT', 'ECFP4', 'MACCS']:
        # Read the fingerprints
        fps = np.load(f'data/fps/{fp}/{name}.npy', mmap_mode='r')

        # Read the similarity matrix
        sim = np.load(f'data/sim_matrix/{fp}/{name}.npy', mmap_mode='r')

        # Calculate the Taylor sum SALI matrix
        ts_sali = ts_sali_matrix(props_diffs, sim, term = 3)

        # Get the average ts_sali
        avg_ts_sali = np.mean(ts_sali)

        # Get the iCliff global
        iCliff_global = calculate_iCliff(fps, props)

        entry = [name, fp, iCliff_global, avg_ts_sali]
        print(entry)
        entries.append(entry)
        

# Save the results
df = pd.DataFrame(entries, columns = ['name', 'fp', 'iCliff', 'ts_sali_avg'])
df.to_csv(f'results/TS_SALI_iCliff.csv', index = False)
