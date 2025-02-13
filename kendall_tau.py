import numpy as np
from scipy.stats import kendalltau
from iCliff.iCliff import ts_sali_matrix, sali_matrix
import glob
import pandas as pd

entries = []
for file in glob.glob('/blue/rmirandaquintana/klopezperez/ecliffs/pair_matrices/prop_diffs/CHEMBL*'):
    # Get the name
    name = file.split('/')[-1].split('.')[0]

    # Read the property matrix
    props = np.load(file)

    for fp in ['RDKIT', 'ECFP', 'MACCS']:
        # Read the similarity matrix
        sim = np.load(f'/blue/rmirandaquintana/klopezperez/ecliffs/pair_matrices/JT/{fp}/{name}_{fp}_JT.npy')

        # Calculate the SALI matrix
        sali = sali_matrix(props, sim)

        for term in [1, 2, 3]:
            # Calculate the Taylor sum SALI matrix
            ts_sali = ts_sali_matrix(props, sim, term)

            # Flatten arrays to calculate KT
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
df.to_csv('kendall_tau_.csv', index = False)
