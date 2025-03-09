import numpy as np
import glob as glob
import matplotlib.pyplot as plt

# Read all the ts_sali matrices
ts_sali_files = glob.glob('data/ts_sali_matrix/ECFP4/*.npy')

ts_sali_90 = 0
ts_sali_95 = 0
ts_sali_99 = 0

# Create a histogram for the matrices values
for file in ts_sali_files:
    ts_sali = np.load(file)

    ts_sali_90 += np.percentile(ts_sali, 90)
    ts_sali_95 += np.percentile(ts_sali, 95)
    ts_sali_99 += np.percentile(ts_sali, 99)

    plt.hist(ts_sali.flatten(), bins=100)
    #plt.title(file)
    plt.xlabel('TS SALI')
    plt.ylabel('Number of occurrences')
    plt.tight_layout()
    plt.savefig(f'results/plots/{file.split("/")[-1].split(".")[0]}_hist.png')
    
    # Clear the plot
    plt.clf()

print(ts_sali_90/30)
print(ts_sali_95/30)
print(ts_sali_99/30)
