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

        if action == "hit" or action == 0:
            # Add the card value to the player's score
            self._player_score += drawCard()

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
            while self._dealer_score < 17 and self._dealer_score > 0:
                # Dealer draws a card              
                self._dealer_score += drawCard()
                    
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
            player_reward = 0.0
                    
        # Report back the player's reward and the game state
        if self._game_over:
            self._player_score = 0.0
            self._dealer_score = 0.0
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

def drawCard():
    # Generate magnitude value of card
    next_card_value = random.randint(1, 10)

    # With probability 1/3, the card we drew was red and has negative value
    if random.randint(1,3) < 2:
        next_card_value *= -1
    return next_card_value