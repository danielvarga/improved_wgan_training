import sys
import os
from tensorboard.backend.event_processing import event_accumulator

import re
import numpy as np

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("rootdir", help="Folder containing the logs")
parser.add_argument("z_key", help="Key for z axis")
parser.add_argument("regexp", type=str, help="Only regexp matched folder names will be parsed")
args = parser.parse_args()

rootdir = args.rootdir
z_key = args.z_key
regexp_to_match = args.regexp

keys_from_filepath = ["lambda", "gp", "gs", "lr", "net", "iters", "train", "wd", "lips", "combslopes", "lrd", "aug", "bs", "bn", "dg"]

def represents_float(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False

records = []
n = 0
for folder, subs, files in os.walk(rootdir):
    for filename in files:
        if not filename.startswith("event"):
            continue

        if regexp_to_match is not None:
            if re.search(regexp_to_match, folder) is None:
                continue

        print(folder)
        logfile = os.path.join(folder, filename)
        n+=1

        record = {}
        d={}

        for key in keys_from_filepath:
            match = re.search("\-"+key+"_(?P<val>[a-z0-9\.^\-]+)\-", folder)
            if match:
                val = match.group('val')
                record[key] = float(val) if represents_float(val) else val

        type = "unknown"
        if record['lambda'] == 0 and 'dg' in record  and  record['dg'] == 0:
            type = 'nogp'
        elif record['lambda'] != 0 and record['gp'] == 2:
            type = 'gp'
        elif record['lambda'] == 0 and 'dg' in record and record['dg'] != 0:
            type = 'datagrad'
        elif record['lambda'] == 0 and record['gs'] == 'y':
            type = 'gs'
        record['type'] = type

        ea = event_accumulator.EventAccumulator(logfile)
        ea.Reload()
        zs = ea.Scalars(z_key)
        z_vals = []
        for z in zs:
          z_vals.append(z.value)
        print(z_vals)
        record[z_key] = z_vals[-1]

        print(record[z_key])
        records.append(record)

print(records)

data = pd.DataFrame(records)
ax = sns.boxplot(x="train", y="accuracy", hue="type", data=data, palette="PRGn")
sns.despine(offset=10, trim=True)
ax.set_title("MNIST, train size: 2000, baseline: LeNet (Lecun et al.), 10 runs")
ax.get_figure().savefig("mnist_boxplot.png")



"""
vals = np.array([rec[z_key] for rec in records])
print(list(vals))
print("n", n)
print("mean:", np.mean(vals))
print("std:", np.std(vals))

"""



