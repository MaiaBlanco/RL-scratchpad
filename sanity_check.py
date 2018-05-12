import pickle 
import numpy as np
from plot import *

def msqerr(v1, v2):
	return np.sum(np.square(v1 - v2))/(11*22*2.0)

value = np.load("montecarlo.npy")
value2 = np.einsum('ijk->jki', value)

f = open('mc.pickle', 'rb')
mc = pickle.load(f)

print( msqerr(mc._action_state_values, value2) )

ax = plot_optimal_value_function(mc._action_state_values, mc._action_state_counts, show=False)
plot_optimal_value_function(value2, None, color='r', show=True, ax=ax)
