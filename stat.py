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

keys_from_filepath = ["lambda", "gp", "gs", "lr", "net", "iters", "train", "wd", "lips", "combslopes", "lrd", "aug", "bs", "bn", "dg", "comb"]

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
            match = re.search("\-"+key+"_(?P<val>[a-z0-9\.^\-]+)\-", folder)
            if match:
                val = match.group('val')
                record[key] = float(val) if represents_float(val) else val

        if record['train'] < 500: # to make the results on the plot more visible
            continue

        
                
        if False: # comparing datagrad parameters with batchnorm
            if record['dg'] == 0 or record['bn'] != "y":
                continue
            type = "dg_bn_" + str(record['dg'])
        elif False: # comparing datagrad parameters with dropout
            if record['dg'] == 0 or record['bn'] == "y":
                continue
            type = "dg_do_" + "%05.2f" % record['dg']
        elif False: # comparing datagrad bn with datagrad do
            if record['dg'] == 0:
                continue
            type = "dg_bn_" + str(record['bn'])
        elif False: # comparing unreg dropout and batchnorm
            if record['dg'] != 0 or record['lambda'] != 0 or record['gs'] == 'y':
                continue
            type = "unreg_bn_" + record['bn']
        elif False: # comparing GP with L2 gradient loss bn vs dropout
            if record['dg'] != 0 or record['lambda'] == 0 or record['gs'] != 'n' or record['comb'] != "random" or record['gp'] != 3:
                continue
            type = "gp3a_bn_" + record['bn']
        elif False: # comparing GP with L2 gradient loss with dropout for various lambdas
            if record['dg'] != 0 or record['lambda'] == 0 or record['gs'] != 'n' or record['comb'] != "random" or record['gp'] != 3 or record['bn'] != "n":
                continue
            type = "gp3a_do_" + "%06.4f" % record['lambda']
        elif False: # comparing GP with softmax bn vs dropout
            if record['dg'] != 0 or record['lambda'] == 0 or record['gs'] != 'n' or record['comb'] != "softmax" or record['gp'] != 3:
                continue
            type = "gp4_bn_" + record['bn']
        elif False: # comparing GP with softmax for various lambdas
            if record['dg'] != 0 or record['lambda'] == 0 or record['gs'] != 'n' or record['comb'] != "softmax" or record['gp'] != 3:
                continue
            type = "gp4_" + "%06.4f" % record['lambda']
        elif False: # comparing GP with L2 gradient loss pushing gradients into LIPSCHITZ_TARGET bn vs dropout
            if record['dg'] != 0 or record['lambda'] == 0 or record['gs'] != 'n' or record['comb'] != "random" or record['gp'] != 4:
                continue
            type = "gp3b_bn_" + record['bn']
        elif False: # comparing GP with L2 gradient loss pushing gradients into LIPSCHITZ_TARGET with dropout for various lipschitz_targets
            if record['dg'] != 0 or record['lambda'] == 0 or record['gs'] != 'n' or record['comb'] != "random" or record['gp'] != 4  or record['bn'] != "n":
                continue
            if record['lips'] > 2:
                continue
            type = "gp3b_do_" + "%05.2f" % record['lips']
        elif False: # comparing GP with L2 gradient loss pushing gradients into LIPSCHITZ_TARGET with dropout for LIPS=0.7 various lambdas
            if record['dg'] != 0 or record['lambda'] == 0 or record['gs'] != 'n' or record['comb'] != "random" or record['gp'] != 4  or record['bn'] != "n" or record['lips'] != 0.7:
                continue
            type = "gp3b_do_lips_0.7_" + "%06.4f" % record['lambda']
        elif True: # final comparison
            if record['dg'] == 0 and record['lambda'] == 0 and record['gs'] == 'n' and record['bn'] == 'n':
                type = "1_unreg"
            elif record['dg'] == 10 and record['bn'] == "n":
                type = "2_datagrad"
            elif record['dg'] == 0 and record['lambda'] == 0.01 and record['comb'] == "random" and record['gp'] == 3 and record['bn'] == 'n':
                type = "3a_gp_to_zero"
            elif record['dg'] == 0 and record['lambda'] == 0.01 and record['comb'] == "random" or record['gp'] == 4  and record['bn'] == "n" and record['lips'] == 0.7:
                type = "3b_gp_to_lips"
            elif record['dg'] == 0 and record['lambda'] == 0.1 and record['comb'] == "softmax" and record['gp'] == 3:
                type = "4_softmax"
            else:
                continue

        else:
            type = "unknown"
            if record['lambda'] == 0 and 'dg' in record  and  record['dg'] == 0:
                type = 'nogp'
            elif record['lambda'] != 0 and record['gp'] == 2:
                type = 'gp'
            elif record['lambda'] == 0 and 'dg' in record and record['dg'] != 0:
                type = 'datagrad'
            elif record['lambda'] == 0 and record['gs'] == 'y':
                type = 'gs'
                
        print(folder)
        record['type'] = type

        ea = event_accumulator.EventAccumulator(logfile)
        ea.Reload()
        zs = ea.Scalars(z_key)
        z_vals = []
        max_step = 0
        for z in zs:
            max_step = z.step
            z_vals.append(z.value)
        if max_step < 10000:
            continue
#        print(z_vals)
        record[z_key] = z_vals[-1]
        if record[z_key] < 0.8:
            continue
        
#        print("train {}, type {}, {} {}".format(record['train'], record['type'], z_key, record[z_key]))
        records.append(record)

#print(records)
records = sorted(records, key=lambda k: k['type']) 
data = pd.DataFrame(records)
ax = sns.boxplot(x="train", y="accuracy", hue="type", data=data, palette="PRGn")
sns.despine(offset=10, trim=True)
ax.set_title("MNIST, train size: 2000, baseline: LeNet (Lecun et al.), 10 runs")
ax.get_figure().savefig("mnist_boxplot.png")

# print averages
grouped_data = data.groupby(('train', 'type'))
print grouped_data[z_key].mean()

"""
vals = np.array([rec[z_key] for rec in records])
print(list(vals))
print("n", n)
print("mean:", np.mean(vals))
print("std:", np.std(vals))

"""



