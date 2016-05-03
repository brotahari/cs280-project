import numpy as np
from matplotlib.pylab import *

with open('./indexed_price_diff.txt', 'r') as f:
    indexed_price_diff = np.array(eval(f.read()))

sorted_ipd = np.array(sorted(indexed_price_diff, key = lambda x:x[1]))
indices = indexed_price_diff[:,0]
true_price = indexed_price_diff[:,1]

figure()
plot( indices, log(sorted_ipd[:,1]))
plot( indices, (log(sorted_ipd[:,1] + sorted_ipd[:,2]))) 
figure()
plot( indices,sorted_ipd[:,1])
plot( indices, (sorted_ipd[:,1] + sorted_ipd[:,2])) 
show()
