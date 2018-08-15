import os
import random
from .globals import FMOVE, UNCOV, MINE, NOTHING
import numpy as np

def array2D(row, col, elem):
    return [[ elem for y in range(col)] for l in range(row)]

def symmetries(matrix):
    """
    Generates all symmetries of the given panel by rotation / mirror
    Args:
        matrix: 2D array

    Yield:
        (matrix, int, bool): tuple containing next matrix, number of 90* rotation and T/F if it was mirrored 
    """
    for i in range(4):
        r = np.rot90(matrix, k=i)
        yield (r, i, False)
        yield (np.fliplr(r), i, True)

def symm_coord(r,c, matrix, nr=0, flip=False):
    m = array2D(len(matrix), len(matrix[0]), 0)
    m[r][c] = 1
    m = np.rot90(m, k=nr)
    m = np.fliplr(m) if flip else m
    #return tuple(a for a in np.argwhere(np.array(m)==1)[0])
    for i in range(len(m)):
        for j in range(len(m[0])):
            if m[i][j] == 1:
                return (i,j)

class Board(object):
    def __init__(self, height, width, mines):
        self.m = mines
        self.h = height # rows
        self.w = width  # cols
        self.firstmove = True

        # solution
        self.minefield = array2D(height, width, '' )
        # current player's view
        self.knowledge = array2D(height, width, UNCOV)
        # count of unvisited cells
        self.nUncov = self.h * self.w


    def neighbourhood(self, x, y):
        """
        Yields coordinates (r,c) for neighbours of the cell (x,y)
        """
        l = []
        l.append((x,y-1)) # n
        l.append((x+1,y-1)) # ne
        l.append((x+1,y)) # e
        l.append((x+1,y+1)) # se
        l.append((x,y+1)) # s
        l.append((x-1,y+1)) # sw
        l.append((x-1,y)) # w
        l.append((x-1,y-1)) # nw
        for r,c in l:
            if r in range(self.h) and c in range(self.w):
                yield (r,c)
        

    def __place_mines(self):
        """
        randomly put mines on the minefield
        """
        placed = 0
        while placed < self.m:
            x = random.randint(0, self.h - 1)
            y = random.randint(0, self.w - 1)
            if self.minefield[x][y] is not MINE and self.minefield[x][y] is not FMOVE :
                # first move should never be a mine
                self.minefield[x][y] = MINE
                placed += 1
    
    def __mine_near(self, x, y):
        count = 0
        for r,c in self.neighbourhood(x, y):
            if self.minefield[r][c] is MINE:
                count += 1
        return count 

    def __hints(self):
        for i in range(self.h):
            for j in range(self.w):
                if self.minefield[i][j] is not MINE:
                    mine_near = self.__mine_near(i,j)
                    self.minefield[i][j] = mine_near if mine_near > 0 else NOTHING

    def generate_board(self):
        self.__place_mines()
        self.__hints()

    def update(self, r, c, log=True):
        """
        Select the cell(r, c) and update the board accordingly. 
        If it is the first action, the board is generated and mines and hints are placed 
        such that the cell (r,c) is not a mine.

        Args:
            r (int): row index
            c (int): column index
            log (bool): enable/disable logging info

        Return:
            value of the cell (r, c)
        """
        if self.firstmove:
            # board generated after the first move to 
            # prevent instant lose
            self.minefield[r][c] = FMOVE
            self.generate_board()
            self.firstmove = False
            
        self.knowledge[r][c] = self.minefield[r][c]
        self.nUncov -= 1
        
        return self.minefield[r][c]

    def __str__(self):
        s = ''
        s += '_'* 25 + '\n'
        for row in self.knowledge:
            s += str(row) + '\n'
        s += '_' * 25
        return s
    
    def draw(self, board):
        # add row/col number
        colnums = [i for i in range(self.w)]
        rownums = [i for i in range(self.h)]
        rownums.insert(0, ' ')
        d = []
        i = 0
        for i in range(len(rownums)):
            row = board[i-1] if i > 0 else colnums 
            d.append(list(row))
            d[i].insert(0, rownums[i])
            i += 1
        
        # draw panel
        s = ''
        s += '_'* 25 + '\n'
        for row in d:
            for e in row:
                s += str(e) + '  '
            s += '\n'
        s += '_' * 25
        print(s)



    def win(self):
        count = 0
        for row in self.knowledge:
            for val in row:
                if val == UNCOV:
                    count += 1
                    if count > self.m:
                        return False
                elif val == MINE:
                    return False 
        return count == self.m

    def clone(self):
        b = Board(self.h, self.w, self.m)
        b.minefield = [list(col) for col in self.minefield]
        b.knowledge = [list(col) for col in self.knowledge]
        b.nUncov = self.nUncov
        b.firstmove = self.firstmove
        return b
            
