import easy21
import TD
from plot import *
import pickle 


# Run TemporalDifference Simulation

RERUN = True

if RERUN:
	td = TD.TemporalDifference21(N0=100.0, n_steps=2)
	for i in range(100000):
		td.run_episode()
		# if i % 10000 == 0:
		# 	print(td._action_state_values[0:2,:,:])
		# 	td.plot_optimal_value_function()
	f = open('td.pickle', 'wb')
	pickle.dump(td, f)
else:
	f = open('td.pickle', 'rb')
	td = pickle.load(f)

plot_optimal_value_function(td._action_state_values, td._action_state_counts, show=True)