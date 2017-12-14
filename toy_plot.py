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

found = False
xs = []
ys = []
dgs = []
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
            match = re.search("dg-(?P<val>([^\_]+|\_[^\_]+))\_", filename)
            if match:
                dg = float(match.group('val'))
            else:
                print "Could not parse datagrad weight from file {}".format(filename)
                continue
            f = np.load(folder + "/" + filename)
            X_devel = f['x']
            y_devel = f['y']
            xs.append(X_devel)
            ys.append(y_devel)
            dgs.append(dg)

xs = np.array(xs)
ys = np.array(ys)
dgs = np.array(dgs)
p = dgs.argsort()
xs = xs[p]
ys = ys[p]
dgs = dgs[p]

fig = plt.figure()
plt.scatter(X_train, y_train, label="train")

d_min = np.where(dgs==0)[0][0]
d_max = np.where(dgs==70)[0][0]
d_step = 4
for i in range(d_min, d_max, d_step):
    plt.plot(xs[i], ys[i], label=int(dgs[i]))

plt.legend(bbox_to_anchor=(0.95, 1.15))

fig.savefig("ttt.png")
plt.close
