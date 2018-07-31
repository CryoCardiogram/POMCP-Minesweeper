from state import State
from board import Board
from globals import *
import random, os

class AbstractPlayer(object):
    def next_action(self, state):
        pass

class RandomPlayer(AbstractPlayer):
    def next_action(self, state):
        return (random.choice(range(state.board.h)), random.choice(range(state.board.w)) )
"""
class QPlayer(AbstractPlayer):
    def __init__(self, ind):
        self.ind = ind
        ## map state, map action -> value
        self.P = dict()
        if os.listdir('obj'):
            self.P = load_obj("P{}".format(ind))

    def update(self, state, cell):
        actions = self.P.get(state, dict())
        occurence = actions.get(cell, 0)
        occurence += 1
        actions.update({cell: occurence})
        self.P.update({state: actions})

    def train(self, board):
        game_over = False
        state = State(board)
        # first move (first cell is always safe)
        state.probe(1, log=False)
        while not game_over:
            cell = random.choice(state.unknown)
            val = state.probe(cell, log=False)
            if val is MINE:
                game_over = True
            else:
                if len(state.unknown) <= state.mines:
                    #for l in board.view:
                    #    print(l)
                    #print("##")
                    game_over = True
                else :
                    self.update(state, cell) #TODO symetries?
        save_obj(self.P, "P{}".format(self.ind))
    
    def selectCell(self, state): 
        s = self.P.get(state, None )
        if s:
            a = max(s, key=lambda k: s[k])
            print("match")
            return a
        else:
            return random.choice(state.unknown)
"""
 







    
    