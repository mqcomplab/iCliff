import numpy as np
import glob as glob
import matplotlib.pyplot as plt

# Read all the ts_sali matrices
ts_sali_files = glob.glob('plots/*ts_sali.npy')

st10, st5, st1 = [], [], []
# Create a histogram for the matrices values
for file in ts_sali_files:
    ts_sali = np.load(file)
    t10, t5, t1 = np.percentile(ts_sali, 90), np.percentile(ts_sali, 95), np.percentile(ts_sali, 99)
    st10.append(t10)
    st5.append(t5)
    st1.append(t1)

print(np.mean(st10), np.mean(st5), np.mean(st1))
