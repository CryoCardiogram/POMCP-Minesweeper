import random
import os
from .model import State, Observation, Action, Minesweeper
from .board import Board
from .globals import MINE
from abc import ABCMeta, abstractmethod
from mcts.pomcp import search, params
from mdp.history import History
from mdp.pomdp import POMDPAction

class AbstractPlayer(object):
    @abstractmethod
    def next_action(self, state):
        pass

class RandomPlayer(AbstractPlayer):
    def next_action(self, state):
        return (random.choice(range(state.board.h)), random.choice(range(state.board.w)) )

class MCPlayer(AbstractPlayer):
    def __init__(self, max_iter, timeout):
        self.max_iter = max_iter
        params['timeout']= timeout
        params.update({
            'timeout': timeout,
            'gamma' : 1.0, # minesweeper is a finite horizon game
            'epsilon': 0.0,
            'log': 2,
            'K': 2,
            'c':0.5
        })
        self.h = History()
        self.last_action = POMDPAction()
        self.first = True

    def next_action(self, state):
        # init domain knowledge
        if self.first:
            self.dom_kno = Minesweeper(state.board.h, state.board.w, state.board.m)
            #self.first = False
        # update history with last action - observation
        o = Observation(state.board.clone().knowledge, state.board.m)
        self.h.add(self.last_action, o)
        #print(self.h)
        # launch UCT to select next best action based on current history
        a = search(self.h.clone(), self.dom_kno, self.max_iter, clean=self.first)
        if self.first:
            self.first = False
        self.last_action = a
        assert isinstance(a, Action)
        return a.cell


        





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
 







    
    