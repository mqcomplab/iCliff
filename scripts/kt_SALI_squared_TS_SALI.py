import numpy as np
from scipy.stats import kendalltau
from iCliff.iCliff import ts_sali_matrix, sali_matrix, squared_prop_diff_matrix, abs_pair_prop_diff_matrix
import glob
import pandas as pd

entries = []

# Read the files with the precomputed squared property differences
for file in glob.glob('data/props/*.npy'):
    # Get the name
    name = file.split('/')[-1].split('.')[0]

    # Read the properties
    props = np.load(file, mmap_mode='r')

    # Calculate the squared diff property matrix
    squared_diff = squared_prop_diff_matrix(props)

    for fp in ['RDKIT', 'ECFP4', 'MACCS']:
        # Read the precomputed similarity matrix
        sim = np.load(f'data/sim_matrix/{fp}/{name}.npy')

        # Calculate the SALI matrix
        sali = sali_matrix(squared_diff, sim)

        for term in [1, 2, 3]:
            # Calculate the Taylor sum SALI matrix
            ts_sali = ts_sali_matrix(squared_diff, sim, term)

            # Flatten arrays to calculate KT, excluding NaNs from the analysis
            flat_sali = sali.flatten()
            non_nan_positions = np.argwhere(~np.isnan(flat_sali))
            flat_sali = flat_sali[non_nan_positions]
            flat_ts_sali = ts_sali.flatten()
            flat_ts_sali = flat_ts_sali[non_nan_positions]

            # Calculate the Kendall Tau
            kt_ts = kendalltau(flat_sali, flat_ts_sali)[0]

            print(f"{name} kendall: {kt_ts}")
            entries.append([name, fp, term, kt_ts])

# Save the results
df = pd.DataFrame(entries, columns = ['name', 'fp', 'term', 'kt'])
df.to_csv('results/kt_SALI_squared_TS_SALI.csv', index = False)
