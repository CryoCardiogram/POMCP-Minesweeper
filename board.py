import os
import random
from globals import *

def array2D(row, col, elem):
    return [[ elem for y in range(col)] for l in range(row)]

class Board(object):
    def __init__(self, height, width, mines):
        self.m = mines
        self.h = height # rows
        self.w = width  # cols
        self.firstmove = True
        
        # solution
        self.reveal = array2D(width, height, '' )
        # current player's view
        self.view = array2D(width, height, UNCOV)

    def neighbourhood(self, x, y, reveal=False):
        """
        Yields neighbours of a cell for the view (default)
        of the revealed version of the board 
        """
        l = []
        b = self.reveal if reveal else self.view
        l.append((x,y-1)) # n
        l.append((x+1,y-1)) # ne
        l.append((x+1,y)) # e
        l.append((x+1,y+1)) # se
        l.append((x,y+1)) # s
        l.append((x-1,y+1)) # sw
        l.append((x-1,y)) # w
        l.append((x-1,y-1)) # nw
        for X,Y in l:
            if Y in range(self.h) and X in range(self.w):
                yield b[X][Y]
        

    def __place_mines(self):
        """
        randomly put mines on the minefield
        """
        placed = 0
        while placed < self.m:
            x = random.randint(0, self.w - 1)
            y = random.randint(0, self.h - 1)
            if self.reveal[x][y] is not MINE and self.reveal[x][y] is not FMOVE :
                # first move should never be a mine
                self.reveal[x][y] = MINE
                placed += 1
    
    def __mine_near(self, x, y):
        count = 0
        for label in self.neighbourhood(x, y, reveal=True):
            if label is MINE:
                count += 1
        return count 

    def __hints(self):
        for i in range(self.w):
            for j in range(self.h):
                if self.reveal[i][j] is not MINE:
                    mine_near = self.__mine_near(i,j)
                    self.reveal[i][j] = mine_near if mine_near > 0 else NOTHING

    def generate_board(self):
        self.__place_mines()
        self.__hints()

    def update(self, x, y, log=True):
        if self.firstmove:
            # board generated after the first move to 
            # prevent instant lose
            self.reveal[x][y] = FMOVE
            self.generate_board()
            self.firstmove = False
            
        self.view[x][y] = self.reveal[x][y]
        if log:
            print("_" * 25),
            for r in self.view:
                print(r)
            print("_" * 25),


