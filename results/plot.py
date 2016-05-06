import json
import numpy as np
from matplotlib import pyplot as plt

with open("./indexed_price_diff_reg1.json") as jf:
    data = np.array(json.load(jf))

indices = data[:,0]
data = np.array( sorted(data, key = lambda x:x[1]) )
true_price = data[:,1]
pred_price = data[:,1] + data[:,2]

with open("./indexed_price_diff_log1.json") as jf:
    data = np.array(json.load(jf))
data = np.array( sorted(data, key = lambda x:x[1]) )
true_log_price = data[:,1]
pred_log_price = data[:,1] + data[:,2]

plt.figure()
plt.scatter( true_price, pred_price, s=1)
plt.figure()
plt.scatter( true_log_price, pred_log_price, s=1)

plt.figure()
plt.plot(indices, true_price)
plt.plot(indices, pred_price)
plt.figure()
plt.plot(indices, true_log_price)
plt.plot(indices, pred_log_price)
plt.show() 
