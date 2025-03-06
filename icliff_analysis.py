import numpy as np
from iCliff.iCliff import calculate_iCliff
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

    # Read if its an AC or not
    ac = np.array(data['cliff']) 

    del data

    # Convert and normalize the properties
    props = np.array(props)/10**9
    props = -np.log(props)

    # Normalize the properties
    props = (props - np.min(props))/(np.max(props) - np.min(props))

    # Calculate the iCliff value
    iCliff = calculate_iCliff(fps, props)

    # Select only the fingeprints that are not AC
    fps_no_AC = fps[ac == 0]
    fps_AC = fps[ac == 1]

    # Select only the properties that are not AC
    props_no_AC = props[ac == 0]
    props_AC = props[ac == 1]    

    # Calculate the iCliff value
    iCliff_no_AC = calculate_iCliff(fps, props)
    iCliff_AC = calculate_iCliff(fps, props)

    entry = [name, iCliff, iCliff_no_AC, iCliff_AC, len(fps), len(fps_no_AC), len(fps_AC)]
    print(entry[:4])

    entries.append(entry)
    

# Save the results
df = pd.DataFrame(entries, columns = ['name', 'iCliff', 'iCliff_no_AC', 'iCliff_AC', 'n', 'n_no_AC', 'n_AC'])
df.to_csv('iCliff_francesca.csv', index=False)
