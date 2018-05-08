import easy21
import pickle 


# Run Time-Difference Simulation

RERUN = True

if RERUN:
	td = easy21.TimeDifference21(N0=1000.0, n_steps=0)
	for i in range(1000000):
		td.run_episode()
		# if i % 1000 == 0:
		# 	print(td._action_state_values[0:2,:,:])
		# 	td.plot_optimal_value_function()
	f = open('td.pickle', 'wb')
	pickle.dump(td, f)
else:
	f = open('td.pickle', 'rb')
	td = pickle.load(f)

td.plot_optimal_value_function()