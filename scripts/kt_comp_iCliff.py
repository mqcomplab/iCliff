import numpy as np
from iCliff.iCliff import calculate_comp_iCliff, ts_sali_matrix
import pandas as pd
import glob
from scipy.stats import kendalltau

fp_type = 'ECFP4'
kendalls = []
for file in glob.glob('data/props/*.npy'):
    # Get the name
    name = file.split('/')[-1].split('.')[0]

    # Read the props
    props = np.load(file, mmap_mode='r')

    # Read the fingerprints
    fps = np.load(f'data/fps/{fp_type}/{name}.npy', mmap_mode='r')

    # Read the ts_sali matrix
    ts_sali = np.load(f'data/ts_sali_matrix/{fp_type}/{name}.npy', mmap_mode='r')

    # Calculate the iCliff values
    iCliff = calculate_comp_iCliff(fps, props)

    # Get the ts_sali sum
    ts_sali_sum = np.sum(ts_sali, axis = 0)

    # Get the ranking correlation for the ts_sali sum and the iCliff values
    kendalls.append([name, kendalltau(ts_sali_sum, 1 - iCliff)[0]])
    

# Save the results
df = pd.DataFrame(kendalls, columns = ['name', 'kendalltau'])
df.to_csv('results/kt_comp_iCliff.csv', index=False)
