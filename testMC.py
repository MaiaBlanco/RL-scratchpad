import easy21
import pickle 


# Run Monte-Carlo Simulation

RERUN = False

if RERUN:
	mc = easy21.MonteCarlo21(N0=1000.0, every_visit=True)
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

mc.plot_optimal_value_function()