import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np

def plot_optimal_value_function(Q, C, show=False, color='b', ax=None):
    state_values = np.zeros((10, 21))
    # Note loops are one-off from corresponding state scores due to 0-based indexing.
    for d_score in range(10):
        for p_score in range(21):
            opt_val = np.amax( Q[d_score, p_score, :] )
            state_values[d_score][p_score] = opt_val
    fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')
    X = range(10)
    Y = range(21)
    X, Y = np.meshgrid(Y, X)
    print(X.shape)
    print(Y.shape)
    print(state_values.shape)
    ax.plot_surface(X+1, Y+1, state_values, color=color)
    plt.title("V*(s) = max_a Q*(s,a)")
    plt.xlabel("Player Score")
    plt.ylabel("Dealer Starting Card (Initial Score)")
    ax.set_zlabel("Value of state given greedy action selection.")
    if show:
        plt.hold(False)
        plt.show()
    else:
        plt.hold(True)
        return ax