# HEADERS
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from easy21 import *

"""
This class implements a framework for performing monte-carlo episodes of easy21.
"""
class MonteCarlo21:
    def __init__(self, N0=100.0, every_visit=True):
        # Init a game instance
        self._game = Easy21()

        # Set initial N for computation of epsilon parameter used with epsilon-greedy
        # policy (default policy). 
        self._N0 = float(N0)

        # Plus one oversizes state space to accomodate for single (-1, -1) terminal state.
        self._action_state_counts = np.zeros((11, 22, 2))
        self._action_state_values = np.zeros((11, 22, 2))

        # Flag for every-visit vs first visit counting:
        self._every_visit = every_visit 


    def run_episode(self):
        '''
        Run a game to completion. When the game is over, update the accumulated returns
        for each state, action pair based on the cumulative record of rewards and state trace.
        '''
        # List to keep track of seen states per episode for first-visit monte carlo
        state_actions = []
        # Rewards trace:
        rewards = []
        Q = self._action_state_values
        C = self._action_state_counts

        # Start the game:
        self._game.reset()
        current_state = self._game.get_state()

        while not self._game._game_over:
            # Subtract one to use scores as indices.
            d_score, p_score = [ x-1 for x in current_state ]

            # Compute epsilon for this iteration:
            N0 = self._N0
            # Compute total number of visits to current state over all actions:
            N_st = float( np.sum(C[d_score, p_score, :]) )
            epsilon = N0/(N0 + N_st)

            # Apply epsilon randomness:
            r = random.random()
            if r < epsilon:
                best_action = random.randint(0,1)
            else:
                # Select the best action given what we know of the state-action estimates:
                best_action = np.argmax(Q[d_score, p_score, :])

            # Update state-action trace for this episode
            state_actions.append( (current_state, best_action) )
            # Take action
            reward, current_state, _ = self._game.player_plays( best_action )
            # Update rewards trace for this episode
            rewards.append(reward)


        # Update value estimates now that episode has concluded:
        seen_states = set()
        for index, state_action in enumerate(state_actions):
            state, action = state_action
            d_score, p_score = [ x-1 for x in state ]
            old_value = Q[d_score, p_score, action]

            # Perform every-visit or first-visit accounting by skipping a previously seen
            # state only if we are doing first-visit:
            if self._every_visit or state not in seen_states:
                # Update state-action count:
                C[d_score, p_score, action] += 1
                count = float(C[d_score, p_score, action])
                # Cumulative sum on rewards:
                cumulative_return = np.sum( rewards[index:] )
                Q[d_score, p_score, action] = old_value + (cumulative_return - old_value) / count
                seen_states.add(state)