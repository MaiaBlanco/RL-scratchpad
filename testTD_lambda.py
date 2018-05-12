import easy21
import TD_lambda as TD
from plot import *
import pickle 


def msqerr(v1, v2):
	return np.sum(np.square(v1 - v2))/(11*22*2.0)

ITERS = 100000
K = 11
with open('mc.pickle', 'rb') as f:
	mc_vals = pickle.load(f)._action_state_values
td_l_vals = []
msqs = np.zeros((K, ITERS))
for k in range(K):
	lambd = 0.1*k
	td = TD.TemporalDifference21_lambd(N0=100.0, lambd=lambd)
	for i in range(ITERS):
		td.run_episode()
		msqs[k,i] = msqerr(mc_vals, td._action_state_values)
		td_l_vals.append( (td._action_state_values, td._action_state_counts) )
	if lambd == 0.0 or lambd == 1.0:
		plt.plot(msqs[k,:], label="Lambda {}".format(lambd))
plt.title("MSQ Err over Iterations")
plt.legend()
plt.ylabel("Mean Squared Error")
plt.xlabel("Episode Number")
plt.show()

plt.plot(np.arange(K)*0.1, msqs[:,-1])
plt.title("MSQ Err over Lambda")
plt.xlabel("Lambda")
plt.ylabel("MSQ Err")
plt.show()

# Lambda = 0
Q, C = td_l_vals[0]
plot_optimal_value_function(Q, C, show=False)
plt.title("Lambda = 0")
plt.hold(False)
plt.show()

# Lambda = 1
Q, C = td_l_vals[-1]
plot_optimal_value_function(Q, C, show=False)
plt.title("Lambda = 1")
plt.hold(False)
plt.show()


print(msqerr(td_l_vals[0][0], td_l_vals[-1][0]))