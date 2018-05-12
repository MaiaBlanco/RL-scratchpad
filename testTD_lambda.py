import easy21
import TD_lambda as TD
from plot import *
import pickle 


def msqerr(v1, v2):
	return np.linalg.norm(v1 - v2)/(11*22*2.0)

K = 2
with open('mc.pickle', 'rb') as f:
	mc_vals = pickle.load(f)._action_state_values
td_l_vals = []
msqs = np.zeros((K, 100000))
for k in range(K):
	lambd = 1*k
	td = TD.TemporalDifference21_lambd(N0=100.0, lambd=lambd)
	for i in range(100000):
		td.run_episode()
		msqs[k,i] = msqerr(mc_vals, td._action_state_values)
		td_l_vals.append( (td._action_state_values, td._action_state_counts) )
	if k == 0 or k == K-1:
		plt.plot(msqs[k,:], label="Lambda {}".format(k*1.0))
plt.title("MSQ Err over Lambda".format(lambd))
plt.legend()
plt.ylabel("Mean Squared Error")
plt.xlabel("Episode Number")
plt.show()
# plt.plot(np.arange(K)*0.1, msqs[:,-1])
# plt.xlabel("Lambda")
# plt.ylabel("MSQ Err")
# plt.show()

# Lambda = 0
Q, C = td_l_vals[0]
plot_optimal_value_function(Q, C, show=True)

# Lambda = 1
Q, C = td_l_vals[-1]
plot_optimal_value_function(Q, C, show=True)