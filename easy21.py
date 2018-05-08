"""
Author: Mark Blanco
Date: 4/6/18
Description:
Functions and classes for David Silver's Easy21 Reinforcement Learning homework assignment.
Org: Carnegie Mellon University
"""

"""
Note to self on code style:
Using Cap-camel case for class names and camel case for standalone functions.
Using underscore_lowercase names for class methods and variables.
If a variable is a member of a class, it is pre-prended by an underscore.
"""

# HEADERS
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

"""
This class implements the SARSA(lambda) reinforcement learning frea
"""
class TemporalDifference:
    '''
    n_steps represents the number of steps ahead that TD learning will use to update state-action value estimations.
    If equal to 'lambda', TD-lambda is used. Otherwise the learning range is exactly n_steps.
    '''
    # TODO: currently only does 0-step TD, not variable step. Fix this.
    def __init__(self, N0=100, n_steps=0):
        # init an interal game instance:
        self._game = Easy21();






"""
This class implements a framework for performing monte-carlo episodes of easy21.
"""
class MonteCarlo21:
    def __init__(self, N0=100, every_visit=True):
        # Init a game instance
        self._game = Easy21()

        # Set initial N for computation of epsilon parameter used with epsilon-greedy
        # policy (default policy). 
        self._N0 = N0

        # self._action_state_returns = np.zeros((10, 21, 2))
        self._action_state_counts = np.zeros((10, 21, 2))
        self._action_state_values = np.zeros((10, 21, 2))

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
        V = self._action_state_values
        # Q = self._action_state_returns
        C = self._action_state_counts
        seen_states = set()
        total_reward = 0.0

        # Start the game:
        self._game.reset()
        iters = 0
        while not self._game._game_over:
            # Get the current state and check if we've seen it (ever):
            current_state = self._game.get_state()
            d_score, p_score = current_state
            # Subtract one to use scores as indices.
            d_score -= 1
            p_score -= 1

            # Compute epsilon for this iteration:
            N0 = self._N0
            # Compute total number of visits to current state over all actions:
            N_st = np.sum(C[d_score, p_score, :])
            epsilon = N0/float(N0 + N_st)

            # Apply epsilon randomness:
            r = random.random()
            if r < epsilon:
                best_action = random.randint(0,len(V[d_score, p_score, :])-1)
            else:
                # Select the best action given what we know of the state-action estimates:
                best_action = np.argmax(V[d_score, p_score, :])

            # Take action
            reward, _, _ = self._game.player_plays( best_action )
            # Update rewards trace for this episode
            rewards.append(reward)
            total_reward += reward
            # Update state-action trace for this episode
            state_actions.append( (current_state, best_action) )

            if self._every_visit or current_state not in seen_states:
                C[d_score, p_score, best_action] += 1

            # Bookkeeping
            iters += 1 
            seen_states.add(current_state)

        # Update value estimates now that episode has concluded:
        seen_states = set()
        for index, state_action in enumerate(state_actions):
            state, action = state_action
            d_score, p_score = state
            d_score -= 1
            p_score -= 1
            old_value = V[d_score, p_score, action]
            count = C[d_score, p_score, action]

            # Perform every-visit or first-visit accounting by skipping a previously seen
            # state only if we are doing first-visit:
            if self._every_visit or state not in seen_states:
                # Cumulative sum on rewards:
                new_cumulative_return = total_reward #np.sum( rewards[index:] )
                V[d_score, p_score, action] = old_value + (new_cumulative_return - old_value) / count
                seen_states.add(state)


    def plot_optimal_value_function(self):
        # Q = self._action_state_returns
        C = self._action_state_counts
        V = self._action_state_values
        state_values = np.zeros((10, 21))
        # Note loops are technically one-off from actual state scores due to 0-based indexing.
        for d_score in range(10):
            for p_score in range(21):
                opt_val = np.max( V[d_score, p_score, :] )
                # count = np.sum( C[d_score, p_score, :] )
                # if (count <= 0):
                #     state_values[d_score][p_score] = 0.0
                #else:
                state_values[d_score][p_score] = opt_val #/ count
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X = range(10)
        Y = range(21)
        X, Y = np.meshgrid(Y, X)
        print(X.shape)
        print(Y.shape)
        print(state_values.shape)
        ax.plot_wireframe(X+1, Y+1, state_values)
        plt.title("V*(s) = max_a Q*(s,a)")
        plt.xlabel("Player Score")
        plt.ylabel("Dealer Starting Card (Initial Score)")
        ax.set_zlabel("Value of state given greedy action selection.")
        plt.show()


""" 
This function implements naive easy 21 agent action selection. 
"""
def naiveEasy21Action(current_score, threshold):
    if current_score < threshold:
        return "hit"
    else:
        return "stick"

"""
Epsilon-greedy approach will find the argmax action given a list of estimated values over
all possible actions (the given state is implicit).
The policy selects the maximum value with probability 1-epsilon, or otherwise selects a
random action.
"""
def epsilonGreedy21(est_values, epsilon):
    r = random.random(0,1)
    if r > epsilon:
        action = np.argmax(est_values)
    else:
        action = random.randint( 0,len(est_values)-1 )
    return action


""" 
This class is meant to implement the easy21 statespace. The agent playing the game
'hit' (draw a card) or 'stick' (nothing) in response to the current sum of cards that 
the dealer and the player have.
Cards are either red or black, and have a value between 1 and 10. Black cards are worth
a positive number of points and red cards are worth negative.
The probability of drawing a black card is 2/3, and the probability of red is 1/3.
Each number is uniformly likely, and cards are sampled from an infinitely replenished
deck (so, with replacement). Lastly, there are no aces (1s are 1s, and cannot be used 
as 10s).
When the game begins, each player is automatically supplied with one black card.
The dealer's strategy is baked into the game model, and therefore into this class:
stick on a sum of 17 or greater, else hit. Note that the dealer only starts to play after
the player 'sticks', which will only happen once because once the player 'sticks' there is 
no going back.
If either player exceeds a sum of 21 or goes below 1, then the game ends and the other 
player wins. When both players (agent and dealer) stick, the game also ends and the
player with the highest score (below 22) wins.
"""
class Easy21:
    def __init__(self):
        self.setup()

    def player_plays(self, action):
        '''
        Action is a string with value 'stick' or 'hit'
        Return value is a tuple of the next state and the revard the player 
        received for the selected action.
        The next state is a list of: [dealer score, player score, game status]
        '''
        # Initialize player reward to 0
        player_reward = 0

        if action == "hit" or action == 0:
            # Generate magnitude value of card
            next_card_value = random.randint(1, 10)

            # With probability 1/3, the card we drew was red and has negative value
            if random.randint(1,3) < 2:
                next_card_value *= -1
            
            # Add the card value to the player's score
            self._player_score += next_card_value

            # Check if the player has gone bust (score over 21 or less than 1)
            if self._player_score > 21 or self._player_score < 1:
                self._game_over = True
                player_reward = -1.0
            else:
                player_reward = 0.0

        
        # Otherwise continue the game by having the dealer play
        elif action == "stick" or action == 1:
            # Dealer functions as a part of the environment and will play ("hit")
            # until either goes bust or score goes above 17
            # "hit" condition:
            while self._dealer_score < 17 and self._dealer_score > 1:
                # Dealer draws a card
                next_card_value = random.randint(1,10)
                    
                if random.randint(1,3) <= 1:
                    next_card_value *= -1
                    
                self._dealer_score += next_card_value
                    
            # Check the dealer's status.
            # If the dealer busts, then the game ends.
            # Otherwise it may be a draw or the player may win.
            self._game_over = True
            if self._dealer_score > 21 or self._dealer_score < 1:
                player_reward = 1.0
            elif self._dealer_score > self._player_score:
                player_reward = -1.0
            elif self._dealer_score < self._player_score:
                player_reward = 1.0
            else:
                player_reward = 0.0

        # If the action is invalid, return without any changes
        else:
            print("Unknown action: {}. Valid actions are 'stick' or 'hit'".format(action))
                    
        # Report back the player's reward and the game state
        state = self.get_state() 
        return ( player_reward, state, self._game_over )
    
    def get_state(self):
        return (self._dealer_score, self._player_score)

    """
    Reset the game to start anew.
    """
    def reset(self):
        self.setup()

    def setup(self):
        # Give each player a black card with value 1-10. Because the 1st card is always
        # black (positive), just set the sum directly.
        self._dealer_score = random.randint(1,10)
        self._player_score = random.randint(1,10)
        
        # Set the game state to non-terminal:
        self._game_over = False

