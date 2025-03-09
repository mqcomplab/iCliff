import numpy as np
from iCliff.iCliff import calculate_comp_iCliff, sali_analysis
import glob

fp_type = 'ECFP4'
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

    # Do the sali analysis
    df = sali_analysis(ts_sali, iCliff, f'results/jaccard/{name}.csv')
