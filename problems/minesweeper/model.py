from problems.minesweeper.board import Board
import random
import math
from problems.minesweeper.globals import UNCOV, MINE, NOTHING
from mdp.pomdp import POMDPState, POMDPObservation, POMDPAction, DecisionProcess

class Observation(POMDPObservation):
    """
    Observations for Minesweeper consist of the knowledge matrix
    """
    def __init__(self, knowledge, mines):
        self.K = knowledge
        self.m = mines
    
    def available_actions(self, h = None):
        if self.is_terminal():
            return 
            yield   # to avoid TypeError

        for row in range(len(self.K)):
            for col in range(len(self.K[0])):
                val = self.K[row][col]
                a = Action(row, col)
                if val == UNCOV and not (h and a in h.actions):
                    yield a
    
    def __eq__(self, oth):
        for i in range(len(self.K)):
            if self.K[i] != oth.K[i]:
                return False
        return True
    
    def __hash__(self):
        t = tuple([ tuple(row) for row in self.K ])
        return hash(t)
    
    def __str__(self):
        s = ''
        for l in self.K:
            s += str(l)
            s += '\n'
        return s 
    
    def is_terminal(self):
        count = 0
        for row in self.K:
            for val in row:
                if val == MINE:
                    return True # loss
                count += 1 if val == UNCOV else 0
        return count == self.m
    
    def __is_start_obs(self):
        for row in self.K:
            for e in row:
                if e != UNCOV:
                    return False
        return True
    
    def __is_corner_move(self, h ,a):
        # first move should be corners, to take advantage
        # of the fact that the first move is always safe
        H = len(self.K)
        W = len(self.K[0])
        corners = {Action(0, 0), Action(H-1, 0), Action(H-1, W-1), Action(0, W-1)}
        return self.__is_start_obs() and a in corners

    def V_init(self, h , a):
        if self.__is_corner_move(h, a):
            print("first move in corner")
            return 10
        return 0
    
    def N_init(self, h, a ):
        if self.__is_corner_move(h,a):
            return 5
        return 0
        
            
class Action(POMDPAction):
    """
    Actions in Minesweeper consist of the coordinates 
    of the cell to probe.
    """
    def __init__(self, r, c):
        self.cell = (r,c)

    def __str__(self):
        return str(self.cell)

    __repr__ = __str__
    
    def __eq__(self, oth):
        if not isinstance(oth, Action):
            return False
        return self.cell == oth.cell
    
    def __hash__(self):
        return hash(self.cell)
    
    def do_on(self, state):
        assert isinstance(state, State)
        state.probe(self.cell[0], self.cell[1], log=False)
        r = 1.0 if state.is_goal() else 0.0
        return (Observation(state.board.knowledge, state.board.m), r)

class State(POMDPState):
    """
    Representation of a state of the minesweeper in our model, 
    """

    def __init__(self, board):
        assert isinstance(board, Board)
        self.board = board
        self.__tM = tuple([ tuple(row) for row in self.board.minefield ])
        self.__tK = tuple([ tuple(row) for row in self.board.knowledge ])

    def __hash__(self):
        return hash((self.__tM, self.__tK, self.board.m))
    
    def __eq__(self, other):
        return (self.__tM, self.__tK, self.board.m) == (other.__tM, other.__tK, other.board.m)
        
    def is_goal(self):
        return self.board.win()

    def clone(self):
        return State(self.board.clone())

    def probe(self, r, c, log=True):
        if log:
            print((r, c))
        val = self.board.update(r, c, log=log)

        # auto reveal of empty cells
        if val is NOTHING:
            #if log:
            #    print("autoprobe")
            autoprob = [ cell for cell in self.board.neighbourhood(r,c) ]
            done = set()
            while autoprob:
                R, C = autoprob.pop()
                v = self.board.update(R, C, log = False)
                done.add((R,C))
                if v is NOTHING:
                    for cell in self.board.neighbourhood(R, C):
                        if cell not in done:
                            autoprob.append(cell)
        return val


class Minesweeper(DecisionProcess):
    def __init__(self, h, w, m):
        self.h = h
        self.w = w
        self.m = m

    def invigoration(self, B):
        # TODO
        pass
    
    def initial_belief(self):
        return State(Board(self.h, self.w, self.m))
        