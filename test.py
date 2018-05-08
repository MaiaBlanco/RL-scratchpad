import easy21
import pickle 

# mc = easy21.MonteCarlo21(N0=100, every_visit=True)
# for i in range(1000000):
# 	mc.run_episode()
# 	# print(i, ": Episode ended!")
# f = open('mc.pickle', 'wb')
# pickle.dump(mc, f)


f = open('mc.pickle', 'rb')
mc = pickle.load(f)

mc.plot_optimal_value_function()