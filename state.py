from board import Board
import random
import math
from globals import *


class State(object):
    """
    Representation of a state of the game in our model, 
    """
    def __init__(self, board):
        assert isinstance(board, Board)
        self.board = board

    def __hash__(self):
        return hash((tuple(self.board.minefield), tuple(self.board.knowledge), self.board.m))
    
    def __eq__(self, other):
        return (tuple(self.board.minefield), tuple(self.board.knowledge), self.board.m) == (tuple(other.board.known), tuple(other.board.knowledge), other.board.m)

    def probe(self, r, c, log=True):
        if log:
            print("({},{})".format(r, c))
        val = self.board.update(r, c, log=log)

        # auto reveal of empty cells
        if val is NOTHING:
            if log:
                print("autoprobe")
            for R,C in self.board.neighbourhood(r, c):
                self.probe(R, C, log=False)

        return val

        


