import numpy as np
from iCliff.iCliff import calculate_comp_iCliff, calculate_iCliff
import glob
import pandas as pd

fp_type = 'ECFP4'
entries = []
for file in glob.glob('data/props/*.npy'):
    # Get the name
    name = file.split('/')[-1].split('.')[0]

    # Read the props
    props = np.load(file, mmap_mode='r')

    # Read the fingerprints
    fps = np.load(f'data/fps/{fp_type}/{name}.npy', mmap_mode='r')

    # Calculate the iCliff values
    iCliff_comp = calculate_comp_iCliff(fps, props)

    # Find the 10% with lowest iCliff values
    idx = np.argsort(iCliff_comp)
    idx = idx[int(len(idx)*0.1):]

    # Remove that 10% from the dataset
    props_ = props[idx]
    fps_ = fps[idx]

    # Calculate the new iCliff values before and after removing the 10%
    iCliff = calculate_iCliff(fps, props)
    iCliff_ = calculate_iCliff(fps_, props_)

    # Save the results
    entries.append([name, iCliff, iCliff_])

# Save the results
df = pd.DataFrame(entries, columns=['name', 'iCliff', 'iCliff_after'])
df.to_csv('results/iCliff_como_ident.csv', index = False)


