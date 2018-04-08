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



"""
This class implements a framework for performing monte-carlo episodes of easy21.
"""
class MonteCarlo21:
    # TODO: implement both versions of MC: every-visit (default) and episode-visit
    def __init__(self, N0=100):
        # Init a game instance
        self._game = Easy21()

        # Set initial N for computation of epsilon parameter used with epsilon-greedy
        # policy (default policy). 
        self._N0 = N0

        # This dictionary is a lookup table of the estimated action-state values.
        # Each key is the state (in easy21 this is a tuple of the dealer score 
        # and agent score).
        # Each value is a tuple containing the list of the total returns for 
        # 'hit' and 'stick' at indices 0 and 1 of the list followed by the 
        # number of hits and sticks performed for the parent state:
        # self._action_values[] => ([returns_hit, returns_stick], [num_hit, num_stick])
        # The sum of both returns over the sum of both totals should give the value
        # estimate over all actions.
        self._action_values = {}

    def run_episode(self):
        '''
        Run a game to completion. When the game is over, update the accumulated rewards
        for each state, action pair.
        '''
        # TODO: figure out if below comments make sense and are necessary...
        # For each episode, keep a running list of each state we have seen.
        # This way we will only add rewards incurred during the game to the states
        # that we have seen.
        #seen_states = set()
        
        # Start the game:
        self._game.reset()
        game_over = False
        while not game_over:
            # Get the current state and check if we've seen it:
            current_state = self._game.get_state()
            if not current_state in self._action_values:
                self._action_values[current_state] = [[0.0,0.0],[0,0]]
            
            value_count_pairs = list(zip(self._action_values[current_state][0], \
                                                            self._action_values[current_state][1]))
            print(value_count_pairs)
            state_value_counts = [ (x/float(y) if y != 0 else 0.0) for (x,y) in value_count_pairs]
            
            # Select the best action given what we know of the state:
            best_action = np.argmax(state_value_counts)

            # Compute epsilon for this iteration:
            N0 = self._N0
            # Compute total number of visits to current state:
            N_st = np.sum(self._action_values[current_state][1])
            epsilon =  N0/float(N0 + N_st)

            # Apply epsilon randomness:
            r = random.random()
            if r < epsilon:
                best_action = random.randint(0,len(state_value_counts)-1)

            # Take action
            reward, current_state, game_over = self._game.player_plays( best_action )
                
            # Update the accumulated rewards for that state,action pair:
            if not current_state in self._action_values:
                self._action_values[current_state] = [[0.0,0.0],[0,0]]
            
            self._action_values[current_state][0][best_action] += reward
            # Incrememt counter for that state, action pair:
            self._action_values[current_state][1][best_action] += 1 
        
                    
    def plot_optimal_value_function(self):
        state_values = np.zeros((21, 21))
        for d_score in range(22):
            for p_score in range(22):
                state = (d_score, p_score)
                if state in self._action_values:
                    value_count_pairs = zip(self._action_values[state])
                    state_value = np.sum([ (x/y if y != 0 else 0) for x,y in value_count_pairs])
                    state_values[d_score][p_score] = state_value
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3D')
        ax.plot_wireframe(range(22), range(22), state_values)
        ax.show()
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
            if random.uniform(0,1) <= 0.333333:
                next_card_value *= -1
            
            # Add the card value to the player's score
            self._player_score += next_card_value

            # Store that reward for later
            player_reward = next_card_value

            # Check if the player has gone bust (score over 21 or less than 1)
            if self._player_score > 21 or self._player_score < 1:
                self._game_over = True
        
        # Otherwise continue the game by having the dealer play
        elif action == "stick" or action == 1:
            # Dealer functions as a part of the environment and will play ("hit")
            # until either goes bust or score goes above 17
            while not self._game_over:
                # "hit" condition and branch
                if self._dealer_score < 17:
                    # Dealer draws a card
                    next_card_value = random.randint(1,10)
                    
                    if random.uniform(0,1) <= 0.333333:
                        next_card_value *= -1
                    
                    self._dealer_score += next_card_value
                    
                    # Check the dealer's status.
                    # If the dealer busts, then the game ends.
                    if self._dealer_score > 21 or self._dealer_score < 1:
                        self._game_over = True

                else: # Dealer sticks and game ends
                    self.game_over = True

        
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

        # Set the player and dealer action states:
        self._dealer_stick = False
        self._player_stick = False

