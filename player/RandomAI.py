'''
Created on Nov 27, 2016

@author: mjw
'''
from player.Player import Player
from engine.Action import Action
import random
from engine.Baron import Baron
from engine.Handmaid import Handmaid

class RandomAI(Player):
    '''
    An AI for engine testing that makes a random choice for all actions.
    
    Alternately, it is an AI that always takes a random choice.
    '''
    
    numBots = 0

    def __init__(self):
        self.number = RandomAI.numBots
        RandomAI.numBots+=1
    
    def getAction(self, dealtcard, deckSize, gravestate, players):
        # ok it's not totally random, but let's not have the bot be a total fool
        # and just play the handmaid on someone else
        target = self
        if not isinstance(dealtcard, Handmaid):
            while target is self:
                target = random.choice(players)
            
        return Action(self, random.choice((self.hand, dealtcard)), target, Baron)
    
    def notifyOfAction(self, action):
        pass
    
    def __str__(self):
        return "RandomAI"+str(self.number)