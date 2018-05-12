import easy21
import MC
from plot import *
import pickle 


# Run Monte-Carlo Simulation

RERUN = True

if RERUN:
	mc = MC.MonteCarlo21(N0=100.0, every_visit=False)
	for i in range(1000000):
		mc.run_episode()
		# if i % 1000 == 0:
		# 	print(mc._action_state_values[0:2,:,:])
		# 	mc.plot_optimal_value_function()
	f = open('mc.pickle', 'wb')
	pickle.dump(mc, f)
else:
	f = open('mc.pickle', 'rb')
	mc = pickle.load(f)

plot_optimal_value_function(mc._action_state_values, mc._action_state_counts, show=True)