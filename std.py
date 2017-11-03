import numpy as np
import sys

a = np.array(map(float, sys.stdin.readlines()))
print a.mean(), a.std(), np.median(a)
print len(a)

