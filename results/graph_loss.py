import json
import numpy as np
from matplotlib import pylab as plt

with open("./loss.json.1462527460") as f:
    data = np.array( json.load(f) )

iter = data[:,0]
loss = data[:,1]

plt.plot(iter, loss)
plt.show()
