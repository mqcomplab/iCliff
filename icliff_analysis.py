import numpy as np
from iCliff.iCliff import calculate_iCliff, ts_sali_matrix
import pandas as pd
import glob

fp_type = 'ECFP'
entries = []
for file in glob.glob('/blue/rmirandaquintana/klopezperez/ecliffs/fps/CHEMBL*.pkl'):
    # Get the name
    name = file.split('/')[-1].split('.')[0]

    # Read the data
    data = pd.read_pickle(file)

    # Read the properties and fps
    props = data['prop']
    fps = np.array(data[fp_type])

    del data

    # Convert and normalize the properties
    props = np.array(props)/10**9
    props = -np.log(props)

    # Normalize the properties
    props = (props - np.min(props))/(np.max(props) - np.min(props))

    # Load the similarity matrix
    sim_matrix = np.load(f'/blue/rmirandaquintana/klopezperez/ecliffs/pair_matrices/JT/{fp_type}/{name}_{fp_type}_JT.npy')

    # Load the property difference matrix
    prop_diff = np.load(f'/blue/rmirandaquintana/klopezperez/ecliffs/pair_matrices/prop_diffs/{name}.npy')

    # Calculate the ts_sali matrix
    ts_sali = ts_sali_matrix(prop_diff, sim_matrix, 3)

    # Find the positions where you find the top 10% of the ts_sali matrix, think that this is a 2D matrix
    top_10 = np.percentile(ts_sali, 90)

    # Find the positions where you find the top 10% of the ts_sali matrix
    top_10_positions = np.argwhere(ts_sali > top_10)

    # Get the unique indexes
    unique_positions = np.unique(top_10_positions.flatten())

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
df.to_csv('iCliff_francesca.csv', index=False)
