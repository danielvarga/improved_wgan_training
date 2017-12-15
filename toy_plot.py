import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator
from matplotlib import cm

import numpy as np

rootdir = "toy" 
keys = ["dg", "wd", "spect"]
sortkey = "spect"

found = False
records = []
for folder, subs, files in os.walk(rootdir):
    for filename in files:
        if not filename.endswith(".npz"):
            continue
        if filename.endswith("train.npz"):
            if not found:
                f = np.load(folder + "/" + filename)
                X_train = f['x']
                y_train = f['y']
                found=True
        else:
            record = {}
            for key in keys:
                match = re.search(key + "-(?P<val>([^\_]+))(\_|\.npz)", filename)
                if match:
                    value = float(match.group('val'))
                    record[key] = value
                else:
                    print "Could not parse {} weight from file {}".format(key, filename)
                    continue
            f = np.load(folder + "/" + filename)
            record['x'] = f['x']
            record['y'] =  f['y']

            # filtering TODO
            if record['dg'] > 0:
                continue
            records.append(record)

records = sorted(records, key=lambda k: k[sortkey]) 

fig = plt.figure()
plt.scatter(X_train, y_train, label="train")

min = -1000
max = 1000
for record in records:
    min = np.min(record['x'].min(), min)
    max = np.max(record['x'].max(), min)
    plt.plot(record['x'], record['y'], label=record[sortkey])

plt.xlim([min, max])

plt.legend(bbox_to_anchor=(1.12, 1.15))
plt.xlabel('x')
plt.ylabel('f(x) = sin(5x)')
plt.title('The effect of SpectReg on function approximation')


fig.savefig("synthetic_sin_spectreg.pdf")
fig.savefig("synthetic_sin_spectreg.png")
plt.close
