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
        self.known = ["" for i in range(board.m * board.w)]
        # unknown cells numeroted by rows, left to right
        self.unknown = [i for i in range(board.m * board.w)]
        self.mines = board.m

    def __hash__(self):
        return hash((tuple(self.known), tuple(self.unknown), self.mines))
    
    def __eq__(self, other):
        return (tuple(self.known), tuple(self.unknown), self.mines) == (tuple(other.known), tuple(other.unknown), other.mines)

    def NH(self, cell):
        """
        Neighborhood of a cell (flatten coordinate)
        """
        l = []
        l.append(cell - self.board.w) # north
        l.append(cell - self.board.w + 1 )  # north east
        l.append(cell +1)   # east
        l.append(  cell + self.board.w + 1) # south east
        l.append(cell + self.board.w) # south
        l.append(cell + self.board.w - 1 ) # south west 
        l.append(cell - 1) # west 
        l.append(cell - self.board.w -1) # north west 
        for cardinal in l:
            if cardinal in range(self.board.h * self.board.w):
                yield cardinal

    def probe(self, cell, log=True):
        if cell in self.unknown :
            x = math.floor(  (cell-self.board.h) / self.board.w)
            y = (cell-1) % self.board.h
            if log:
                print("({},{})".format(x, y))
            val = self.board.reveal[x][y]
            self.board.update(x, y, log=log)
            self.unknown.remove(cell)
            self.known[cell] = val
            # auto reveal of empty cells
            if val is NOTHING:
                if log:
                    print("autoprob")
                for neighb in self.NH(cell):
                    self.probe(neighb, log=False)

            return val
        else:
            return None
        


