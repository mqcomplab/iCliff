import numpy as np
from iCliff.iCliff import calculate_iCliff
import pandas as pd
import glob

fp_type = 'ECFP4'
entries = []
for file in glob.glob('data/props/*.npy'):
    # Get the name
    name = file.split('/')[-1].split('.')[0]

    # Read the props data
    props = np.load(file, mmap_mode='r')

    # Read the fingerprints data
    fps = np.load(f'data/fps/{fp_type}/{name}.npy', mmap_mode='r')
    
    # Read the ts_sali matrix
    ts_sali = np.load(f'data/ts_sali/{fp_type}/{name}.npy', mmap_mode='r')

    # Find the positions where you find the top 10% of the ts_sali matrix, think that this is a 2D matrix
    top = np.percentile(ts_sali, 99.9)

    # Find the positions where you find the top 10% of the ts_sali matrix
    top_positions = np.argwhere(ts_sali > top)

    # Get the unique indexes
    unique_positions = np.unique(top_positions.flatten())

    # Calculate the iCliff value
    iCliff = calculate_iCliff(fps, props)

    # Select only the fingeprints that are not AC (on the unique positions)
    fps_AC = fps[unique_positions]
    fps_no_AC = np.delete(fps, unique_positions, axis=0)

    # Select only the properties that are not AC (on the unique positions)
    props_AC = props[unique_positions]
    props_no_AC = np.delete(props, unique_positions)

    # Calculate the iCliff value
    iCliff_no_AC = calculate_iCliff(fps_no_AC, props_no_AC)
    iCliff_AC = calculate_iCliff(fps_AC, props_AC)

    entry = [name, iCliff, iCliff_no_AC, iCliff_AC, len(fps), len(fps_no_AC), len(fps_AC)]
    print(entry[:4])

    entries.append(entry)
    

# Save the results
df = pd.DataFrame(entries, columns = ['name', 'iCliff', 'iCliff_no_AC', 'iCliff_AC', 'n', 'n_no_AC', 'n_AC'])
df.to_csv('results/iCliff_analysis.csv', index=False)
