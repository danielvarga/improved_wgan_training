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
parser.add_argument("MAX_STEP", type=int, help="Discard runs that did not run until MAX_STEP iterations")
parser.add_argument("-x_key", default="train", help="Key for x axis")
args = parser.parse_args()

import stat_type

rootdir = args.rootdir
z_key = args.z_key
x_key = args.x_key
regexp_to_match = args.regexp
MAX_STEP = args.MAX_STEP

keys_from_filepath = ["lambda", "gp", "gs", "lr", "net", "iters", "train", "wd", "lips", "combslopes", "lrd", "aug", "bs", "bn", "dg", "comb", "ent", "do"]

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

        logfile = os.path.join(folder, filename)
        n+=1

        record = {}
        d={}

        for key in keys_from_filepath:
            match = re.search("\-"+key+"_(?P<val>([^\-]+|\-[^\-]+))\-", folder)
            if match:
                val = match.group('val')
                record[key] = float(val) if represents_float(val) else val


        if record['bn'] == "y":
            record['reg'] = 'batchnorm'
        elif record['do'] == 1:
            record['reg'] = "none"
        else:
            record['reg'] = "dropout"

        type = stat_type.get_type(record)
        if type == None:
            continue
        else:
            print(folder)
            record['type'] = type

        ea = event_accumulator.EventAccumulator(logfile)
        ea.Reload()
        try:
            zs = ea.Scalars(z_key)
        except:
            continue
        z_vals = []
        max_step = 0
        for z in zs:
            max_step = z.step
            z_vals.append(z.value)
        if max_step < MAX_STEP:
            continue
#        print(z_vals)
        record[z_key] = z_vals[-1]
        
#        print("train {}, type {}, {} {}".format(record['train'], record['type'], z_key, record[z_key]))
        records.append(record)

print(records)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
records = sorted(records, key=lambda k: k['type']) 
data = pd.DataFrame(records)

print(data.groupby("type")['accuracy'].describe())

ax = sns.boxplot(x=x_key, y="accuracy", hue="type", data=data, palette="PRGn")
sns.despine(offset=10, trim=True)
# ax.set_title("MNIST, train size: 2000, baseline: LeNet (Lecun et al.), 10 runs")
ax.get_figure().savefig("mnist_boxplot.png")

# print averages
grouped_data = data.groupby((x_key, 'type'))
print grouped_data[z_key].mean()

"""
vals = np.array([rec[z_key] for rec in records])
print(list(vals))
print("n", n)
print("mean:", np.mean(vals))
print("std:", np.std(vals))

"""



