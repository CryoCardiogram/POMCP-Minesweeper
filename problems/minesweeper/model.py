from problems.minesweeper.board import Board
import random
import math
from problems.minesweeper.globals import UNCOV, MINE, NOTHING
from mdp.pomdp import POMDPState, POMDPObservation, POMDPAction, DecisionProcess
from mcts.pomcp import params

class Observation(POMDPObservation):
    """
    Observations for Minesweeper consist of the knowledge matrix
    """
    def __init__(self, knowledge, mines):
        self.K = knowledge
        self.m = mines
        self.__t = tuple([ tuple(row) for row in self.K ])
    
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
        if not isinstance(oth, Observation):
            return False 
        return self.__t == oth.__t and self.m == oth.m
    
    def __hash__(self):
        t = tuple([ tuple(row) for row in self.K ])
        return hash((t, self.m)) 
    
    def __str__(self):
        s = ''
        for l in self.K:
            s += str(l)
            s += '\n'
        return s 
    
    #__repr__= __str__
    
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
            #print("first move in corner")
            return 1
        return 0
    
    def N_init(self, h, a ):
        if self.__is_corner_move(h,a):
            return 1
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
        init_len = len(state.interior) + len(state.frontier)
        val = state.probe(self.cell[0], self.cell[1], log=False)
        after = len(state.interior) + len(state.frontier)
        # intermediate reward of 1 per probed cell (before landing on a mine)
        r = 0 if val == MINE else after - init_len
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
        # cells of the board are divided in different sets
        # interior: set of covered cells not adjacent to a mine (cells containing NOTHING )
        self.interior = set()
        # frontier: set of covered cells ajacent to at least 1 mine (cells with hints)
        self.frontier = set()
        # fringe: set of uncovered cells adjacent to at least 1 frontier cell
        self.fringe = set()
        # uncovered: set of remaining uncovered cells
        self.uncovs = set()
        for r in range(len(self.board.knowledge)):
            for c in range(len(self.board.knowledge[0])):
                self.uncovs.add((r,c))

    def __hash__(self):
        return hash((self.__tM, self.__tK, self.board.m))
    
    def __eq__(self, other):
        return (self.__tM, self.__tK, self.board.m) == (other.__tM, other.__tK, other.board.m)
        
    def is_goal(self):
        return self.board.win()

    def clone(self):
        s = State(self.board.clone())
        sets = ['interior', 'frontier', 'fringe', 'uncovs']
        for setname in sets:
            setattr(s, setname, set({cell for cell in getattr(self, setname)}) )
        return s
    
    def ___remove_from_uncovs(self, R, C):
        # sets update
        try:
            self.fringe.remove((R,C))
        except KeyError:
            self.uncovs.discard((R,C))   
            self.fringe.discard((R,C))  

    def __update_fringe(self):
        adj = set()
        for r,c in self.frontier:
            adj.update({cell for cell in self.board.neighbourhood(r,c)})

        for cell in adj:
            if cell in self.uncovs:
                self.uncovs.discard(cell)
                self.fringe.add(cell)
        assert self.fringe.isdisjoint(self.uncovs)

    def probe(self, r, c, log=True):
        if log:
            print((r, c))
        val = self.board.update(r, c, log=log)
        self.___remove_from_uncovs(r,c)

        # auto reveal of empty cells
        if val is NOTHING:
            #if log:
            #    print("autoprobe")
            self.interior.add((r,c))
            autoprob = [ cell for cell in self.board.neighbourhood(r,c) ]
            done = set()
            while autoprob:
                R, C = autoprob.pop()
                self.___remove_from_uncovs(R,C)
                v = self.board.update(R, C, log = False)
                done.add((R,C))
                if v is NOTHING:
                    self.interior.add((R,C))
                    for cell in self.board.neighbourhood(R, C):
                        if cell not in done:
                            autoprob.append(cell)
                else:
                    self.frontier.add((R,C))
        else:
            self.frontier.add((r,c))
        self.__update_fringe()
        self.__tM = tuple([ tuple(row) for row in self.board.minefield ])
        self.__tK = tuple([ tuple(row) for row in self.board.knowledge ])
        return val


class Minesweeper(DecisionProcess):
    def __init__(self, h, w, m):
        self.h = h
        self.w = w
        self.m = m

    def __empty_belief(self, B, h):
        # create at least 1 particle that can then be used by the invigoration algorithm,
        # based on the current observation
        pass

    def invigoration(self, B, h):
        assert len(B) > 0, "empty belief: {}".format(h)
        max_to_add = math.floor(1/params['K'] * len(B))
        init_len = len(B)
        tries = 0

        def mine_near(b, x, y):
            count = 0
            for r,c in b.neighbourhood(x, y):
                if b.minefield[r][c] is MINE:
                    count += 1
            return count 

        while len(B) != max_to_add + init_len and tries < 100:
            tries += 1 # try at most 100 times
            rnd = random.choice(tuple(B))
            # we only consider mines in the set of uncovered cells
            uncov_mines = 0
            mines = set()
            uncov_copy = set({cell for cell in rnd.uncovs})
            particle = rnd.clone() # artificial state to add noise in the belief set
            for r,c in uncov_copy:
                # clear the uncovered minefield of our particle
                particle.board.minefield[r][c] = NOTHING
                if rnd.board.minefield[r][c] == MINE:
                    # count the mines within uncovered cells
                    mines.add((r,c))
                    uncov_mines += 1
                
            # we randomly change location of mines in the set of uncovered cells
            new_locations = set()
            while len(new_locations) < uncov_mines:
                rnd_cell = random.choice(tuple(uncov_copy))
                if rnd_cell not in new_locations:
                    uncov_copy.discard(rnd_cell)
                    new_locations.add(rnd_cell)

            # place mines at new location and compute hints 
            for r,c in particle.uncovs:
                if (r,c) in new_locations:
                    particle.board.minefield[r][c] = MINE
                else:
                    mn = mine_near(particle.board, r, c)
                    particle.board.minefield[r][c] = mn if mn > 0 else NOTHING
            
            B.append(particle)
        print("{} state(s) added".format(len(B) - init_len))

    def initial_belief(self):
        return State(Board(self.h, self.w, self.m))
        