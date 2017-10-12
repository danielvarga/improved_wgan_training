import sys
import os
from tensorboard.backend.event_processing import event_accumulator

import re

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("rootdir", help="Folder containing the logs")
parser.add_argument("x_key", help="Key for x axis")
parser.add_argument("y_key", help="Key for y axis")
parser.add_argument("z_key", help="Key for the color axis")
parser.add_argument("--regexp", type=str, help="Only regexp matched folder names will be parsed")
parser.add_argument("--outfile_prefix", type=str, help="Prepends outoup filename with given string")
parser.add_argument("--vmin", type=float, help="Min colored value on the heatmap")
parser.add_argument("--vmax", type=float, help="Max colored value on the heatmap")
args = parser.parse_args()

rootdir = args.rootdir
x_key = args.x_key
y_key = args.y_key
z_key = args.z_key
regexp_to_match = args.regexp
outfile_prefix = args.outfile_prefix

vmin = args.vmin
vmax = args.vmax

records = []

for folder, subs, files in os.walk(rootdir):
    for filename in files:

        if not filename.startswith("event"):
            continue

        if regexp_to_match is not None:
            if re.search(regexp_to_match, folder) is None:
                continue

        print(folder)
        logfile = os.path.join(folder, filename)

        record = {}
        d={}

        matchObj = re.search(x_key + r'_([0-9\.e\-]*)\-', folder, re.M | re.I)
        d[x_key] = matchObj.group(1)

        matchObj = re.search(y_key + r'_([0-9\.e\-]*)\-', folder, re.M | re.I)
        d[y_key] = matchObj.group(1)

        ea = event_accumulator.EventAccumulator(logfile)
        ea.Reload()
        zs = ea.Scalars(z_key)

        record[x_key] = float(d[x_key])
        record[y_key] = float(d[y_key])
        record[z_key] = zs[-1].value
        records.append(record)

df = pd.DataFrame(records)
df2 = df.pivot(y_key, x_key, z_key)
f, ax = plt.subplots(figsize=(16, 9))

if outfile_prefix is not None:
    f.suptitle(outfile_prefix, fontsize=16)

sns_plot = sns.heatmap(df2, annot=True, linewidths=.5, ax=ax, fmt=".3f", vmin=vmin, vmax=vmax)
ax.set_title(z_key)

outfile = "heatmap_{0}_{1}_{2}.png".format(y_key, x_key, z_key)
if outfile_prefix is not None:
    outfile = outfile_prefix + "_" + outfile

sns_plot.get_figure().savefig(outfile)
