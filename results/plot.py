import json
import numpy as np
from matplotlib import pyplot as plt


with open("./indexed_price_diff.json.1462531520") as jf:
    data = np.array(json.load(jf))
indices = data[:,0]
data = np.array( sorted(data, key = lambda x:x[1]) )
true_log_price = data[:,1]
pred_log_price = data[:,1] + data[:,2]

plt.figure()
plt.scatter( true_log_price, pred_log_price, s=1)

print true_log_price[5000//3]
print true_log_price[2*5000//3]

plt.figure()
plt.plot(indices, true_log_price)
plt.plot(indices, pred_log_price)
plt.show() 
