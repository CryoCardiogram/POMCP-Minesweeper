from minesweeper.board import Board
import random
import math
from minesweeper.globals import *
from mdp.pomdp import POMDPState, POMDPObservation, POMDPAction

class Observation(POMDPObservation):
    """
    Observations for Minesweeper consist of the knowledge matrix
    """
    def __init__(self, knowledge):
        self.K = knowledge
    
    def available_actions(self):
        for row in range(len(self.K)):
            for col in range(len(self.K[0])):
                val = self.K[row][col]
                if val == UNCOV:
                    yield Action(row, col)
    
    def __eq__(self, oth):
        for i in range(len(self.K)):
            if self.K[i] != oth.K[i]:
                return False
        return True
    
    def __hash__(self):
        t = tuple([ tuple(row) for row in self.K ])
        hash(t)
    
    def __str__(self):
        s = ''
        for l in self.K:
            s += str(l)
            s += '\n'
        return s 
            
            



class Action(POMDPAction):
    """
    Actions in Minesweeper consist of the coordinates 
    of the cell to probe.
    """
    def __init__(self, r, c):
        self.cell = (r,c)

    def __str__(self):
        return "({},{})".format( self.cell[0], self.cell[1])

    __repr__ = __str__
    
    def __eq__(self, oth):
        return self.cell == oth.cell
    
    def do_on(self, state):
        assert isinstance(state, State)
        state.probe(self.cell[0], self.cell[1], log=False)
        return Observation(state.board.knowledge) 

class State(POMDPState):
    """
    Representation of a state of the minesweeper in our model, 
    """

    def __init__(self, board):
        assert isinstance(board, Board)
        self.board = board

    def __hash__(self):
        return hash((tuple(self.board.minefield), tuple(self.board.knowledge), self.board.m))
    
    def __eq__(self, other):
        return (tuple(self.board.minefield), tuple(self.board.knowledge), self.board.m) == (tuple(other.board.known), tuple(other.board.knowledge), other.board.m)

    def is_goal(self):
        return self.board.win()

    def probe(self, r, c, log=True):
        if log:
            print("({},{})".format(r, c))
        val = self.board.update(r, c, log=log)

        # auto reveal of empty cells
        if val is NOTHING:
            if log:
                print("autoprobe")
            for R,C in self.board.neighbourhood(r, c):
                self.board.update(r,c, log = False)

        return val

        


