import easy21
import TD_lambda as TD
from plot import *
import pickle 


def msqerr(v1, v2):
	return np.linalg.norm(v1 - v2)


with open('mc.pickle', 'rb') as f:
	mc_vals = pickle.load(f)._action_state_values
td_l_vals = []
msqs = np.zeros((11, 1000))
for k in range(11):
	lambd = 0.1*k
	td = TD.TemporalDifference21_lambd(N0=100.0, lambd=lambd)
	for i in range(1000):
		td.run_episode()
		msqs[k,i] = msqerr(mc_vals, td._action_state_values)
		td_l_vals.append( (td._action_state_values, td._action_state_counts) )
	plt.plot(msqs[k,:])
	plt.title("MSQ Err for Lambda = {}".format(0.1*k))
	plt.show()

# Lambda = 0
Q, C = td_l_vals[0]
plot_optimal_value_function(Q, C, show=True)

# Lambda = 1
Q, C = td_l_vals[-1]
plot_optimal_value_function(Q, C, show=True)