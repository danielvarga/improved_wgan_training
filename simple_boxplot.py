import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("bests.tsv", sep="\t")
ax = sns.boxplot(x="model", y="accuracy", hue="model", data=data, palette="PRGn")
sns.despine(offset=10, trim=True)
ax.set_title("MNIST, train size: 2000, baseline: LeNet (Lecun et al.), 10 runs")
ax.get_figure().savefig("mnist_boxplot.png")
