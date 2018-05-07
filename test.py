import easy21

mc = easy21.MonteCarlo21(N0=100, every_visit=True)
for i in range(100000):
	mc.run_episode()
	# print(i, ": Episode ended!")

# mc.plot_expected_value_function()
mc.plot_optimal_value_function()