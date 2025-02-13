import numpy as np
from iCliff.iCliff import calculate_iCliff, ts_sali_max, sali_analysis, ts_sali_matrix, jaccard
import pandas as pd
import glob

fp_type = 'ECFP'
jaccards = []
for file in glob.glob('/blue/rmirandaquintana/klopezperez/ecliffs/fps/CHEMBL*'):
    # Get the name
    name = file.split('/')[-1].split('.')[0]

    # Read the data
    data = pd.read_pickle(f'/blue/rmirandaquintana/klopezperez/ecliffs/fps/{file.split(".")[0]}.pkl')

    # Read the properties and fps
    props = data['prop']
    fps = data[fp_type]

    # Read if its an AC or not
    ac = np.array(data['ciff'])
    n_ac = np.sum(ac) 

    del data

    # Convert and normalize the properties
    props = np.array(props)/10**9
    props = -np.log(props)

    # Normalize the properties
    props = (props - np.min(props))/(np.max(props) - np.min(props))

    # Calculate the iCliff values
    iCliff = calculate_iCliff(props, fps)

    # Calculate the ts_sali
    prop_matrix = np.load(f'/blue/rmirandaquintana/klopezperez/ecliffs/pair_matrices/prop_diffs/CHEMBL*', mmap_mode='r')
    sim_matrix = np.load(f'/blue/rmirandaquintana/klopezperez/ecliffs/pair_matrices/JT/{fp_type}/{file.split(".")[0]}_{fp_type}_JT.npy', mmap_mode='r')

    ts_sali = ts_sali_matrix(props, fps, term=3)

    del prop_matrix, sim_matrix

    # Do the analysis ts_sali vs iCliff
    a = sali_analysis(ts_sali, iCliff, name)
    del a

    # Do the analysis ts_sali_max vs iCliff
    max_ts_sali = ts_sali_max(ts_sali, frac_max=0.2)
    b = sali_analysis(max_ts_sali, iCliff, name)
    del b

    # Do the analysis francesca vs iCliff
    # Get the indices of the 20 lowest iCliff values
    idx = np.argsort(iCliff)[:n_ac]

    # Get the indices where ac is True
    ac_idx = np.where(ac)[0]

    # Calculate the jaccard index
    jaccard_idx = jaccard(idx, ac_idx)
    jaccards.append([name, jaccard_idx])

    print(f"{name} jaccard: {jaccard_idx}")

# Save the results
df = pd.DataFrame(jaccards, columns = ['name', 'jaccard'])