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
parser.add_argument("-type_grouping", default=None, help="Specify how runs are grouped into types. See stat_type.py")
args = parser.parse_args()

import stat_type

rootdir = args.rootdir
z_key = args.z_key
regexp_to_match = args.regexp
MAX_STEP = args.MAX_STEP
x_key = args.x_key
type_grouping = args.type_grouping

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
            # match = re.search("\-"+key+"_(?P<val>([^\-]+|\-[^\-]+))\-", folder)
            match = re.search("\-"+key+"_(?P<val>([0-9].*[0-9]*e\-[0-9]+|[^\-]+|\-[^\-]+))\-", folder) # allow for 3e-6 like expressions
            if match:
                val = match.group('val')
                record[key] = float(val) if represents_float(val) else val


        if record['bn'] == "y":
            record['reg'] = 'batchnorm'
        elif record['do'] == 1:
            record['reg'] = "none"
        else:
            record['reg'] = "dropout"

        type = stat_type.get_type(record, type_grouping)
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
            print max_step
            continue
#        print(z_vals)
        record[z_key] = z_vals[-1]
        
#        print("train {}, type {}, {} {}".format(record['train'], record['type'], z_key, record[z_key]))
        records.append(record)

# print(records)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#mnist_4 ordering:
def mnist_4_ordering(t):
    if t == "DoubleBack":
        return 0
    elif t == "SpectReg":
        return 1
    elif t == "JacReg":
        return 2
    elif t == "Confidence Penalty":
        return 3
    else:
        return 4
# records = sorted(records, key=lambda k: mnist_4_ordering(k['type'])) 
records = sorted(records, key=lambda k: k['type']) 

data = pd.DataFrame(records)
print(data.groupby("type")[z_key].describe())

ax = sns.boxplot(x=x_key, y=z_key, hue="type", data=data, palette="PRGn")
# ax.set(xlabel='Train size',ylabel='Test accuracy')
plt.tight_layout()
sns.despine(offset=10, trim=True)
# ax.set_title("MNIST, train size: 2000, baseline: LeNet (Lecun et al.), 10 runs")
ax.get_figure().savefig("mnist_boxplot.png")
ax.get_figure().savefig("mnist_boxplot.pdf")

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



