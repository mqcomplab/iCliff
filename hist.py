import numpy as np
import glob as glob
import matplotlib.pyplot as plt

# Read all the ts_sali matrices
ts_sali_files = glob.glob('plots/*ts_sali.npy')

# Create a histogram for the matrices values
for file in ts_sali_files:
    ts_sali = np.load(file)
    plt.hist(ts_sali.flatten(), bins=100)
    plt.title(file)
    plt.savefig(f'plots/{file.split("/")[-1].split(".")[0]}_hist.png')


