import numpy as np
from iCliff.iCliff import calculate_iCliff, ts_sali_max, sali_analysis, ts_sali_matrix, jaccard, sali_matrix
import pandas as pd
import glob

fp_type = 'ECFP'
jaccards = []
for file in glob.glob('/blue/rmirandaquintana/klopezperez/ecliffs/fps/CHEMBL*.pkl'):
    # Get the name
    name = file.split('/')[-1].split('.')[0]

    # Read the data
    data = pd.read_pickle(file)

    # Read the properties and fps
    props = data['prop']
    fps = np.array(data[fp_type])

    # Read if its an AC or not
    ac = np.array(data['cliff'])
    n_ac = np.sum(ac) 

    del data

    # Convert and normalize the properties
    props = np.array(props)/10**9
    props = -np.log(props)

    # Normalize the properties
    props = (props - np.min(props))/(np.max(props) - np.min(props))

    # Calculate the iCliff values
    iCliff = calculate_iCliff(fps, props)

    # Calculate the ts_sali
    prop_matrix = np.load(f'/blue/rmirandaquintana/klopezperez/ecliffs/pair_matrices/prop_diffs/{name}.npy', mmap_mode='r')
    sim_matrix = np.load(f'/blue/rmirandaquintana/klopezperez/ecliffs/pair_matrices/JT/{fp_type}/{name}_{fp_type}_JT.npy', mmap_mode='r')

    ts_sali = ts_sali_matrix(prop_matrix, sim_matrix, term=3)


    # Do the analysis ts_sali vs iCliff
    a = sali_analysis(ts_sali, iCliff, 'ts_sali_vs_icliff/' + name + '.csv')
    del a

    # Do the analysis ts_sali_max vs iCliff
    max_ts_sali = ts_sali_max(ts_sali, frac_max=0.2)
    b = sali_analysis(max_ts_sali, iCliff, 'max_ts_sali_vs_icliff/' + name + '.csv')
    del b

    # Do the analysis sali vs iCliff
    sali = sali_matrix(prop_matrix, sim_matrix)
    a = sali_analysis(sali, iCliff, 'sali_vs_icliff/' + name + '.csv')
    del a

    # Do the analysis max_sali vs iCliff
    max_sali = ts_sali_max(sali, frac_max = 0.2)
    a = sali_analysis(max_sali, iCliff, 'max_sali_vs_icliff/' + name + '.csv')
    del a

    del prop_matrix, sim_matrix
    # Do the analysis francesca vs iCliff
    # Get the indices of the 20 lowest iCliff values
    idx = np.argsort(iCliff)[:int(n_ac*1.5)]

    # Get the indices where ac is True
    ac_idx = np.where(ac)[0]

    # Calculate the jaccard index
    jaccard_idx = jaccard(idx, ac_idx)
    jaccards.append([name, jaccard_idx, n_ac])

    print(f"{name} jaccard: {jaccard_idx}")

# Save the results
df = pd.DataFrame(jaccards, columns = ['name', 'jaccard', 'n_ac'])
df.to_csv('jaccards_francesca.csv', index=False)
