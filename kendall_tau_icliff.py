import numpy as np
from iCliff.iCliff import calculate_comp_iCliff, ts_sali_matrix
import pandas as pd
import glob
from scipy.stats import kendalltau

fp_type = 'ECFP'
kendalls = []
for file in glob.glob('/blue/rmirandaquintana/klopezperez/ecliffs/fps/CHEMBL*.pkl'):
    # Get the name
    name = file.split('/')[-1].split('.')[0]

    # Read the data
    data = pd.read_pickle(file)

    # Read the properties and fps
    props = data['prop']
    fps = np.array(data[fp_type])

    # Convert and normalize the properties
    props = np.array(props)/10**9
    props = -np.log(props)

    # Normalize the properties
    props = (props - np.min(props))/(np.max(props) - np.min(props))

    # Calculate the iCliff values
    iCliff = calculate_comp_iCliff(fps, props)

    # Calculate the ts_sali
    prop_matrix = np.load(f'/blue/rmirandaquintana/klopezperez/ecliffs/pair_matrices/prop_diffs/{name}.npy', mmap_mode='r')
    sim_matrix = np.load(f'/blue/rmirandaquintana/klopezperez/ecliffs/pair_matrices/JT/{fp_type}/{name}_{fp_type}_JT.npy', mmap_mode='r')

    ts_sali = ts_sali_matrix(prop_matrix, sim_matrix, term=3)
    ts_sali_sum = np.sum(ts_sali, axis = 0)

    kendalls.append([name, kendalltau(ts_sali_sum, 1 - iCliff)[0]])
    

# Save the results
df = pd.DataFrame(kendalls, columns = ['name', 'kendalltau'])
df.to_csv('kendalls_icliff.csv', index=False)
