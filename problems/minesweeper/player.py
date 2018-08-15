import random
import os
from .model import State, Observation, Action, Minesweeper
from .board import Board, symmetries, symm_coord
from .globals import MINE, load_obj, save_obj
from abc import ABCMeta, abstractmethod
from mcts.pomcp import search, params
from mdp.history import History
from mdp.pomdp import POMDPAction

class AbstractPlayer(metaclass=ABCMeta):
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


class QPlayer(AbstractPlayer):
    """
    This player follows a simplified Q-learning approach. The value function is 
    approximated by a Q-learning algorithm of the following form:
    Q*(s,a) <- (1 - η)Q*(s,a) + η(r)
    where the intermediate reward r is 1 for choosing a non-mined cell and -1
    otherwise. 
    In order to improve information gain, this improved Q-learning agent only focuses on
    correct moves. P(s,a) is a value representing the probability the cell probed by 
    action a on a state s does not contains a mine.
    """
    def __init__(self, ind, sym=True):
        # id
        self.ind = ind
        ## map obs, map action -> value
        self.P = dict()
        if os.listdir('obj'):
            if 'P{}.pkl'.format(ind) in os.listdir('obj'):
                self.P = load_obj("P{}".format(ind))
        self.sym = sym

    def __sym_update(self, state, cell): 
        for k,m in zip(symmetries(state.board.knowledge), symmetries(state.board.minefield)):
            b = Board(len(k), len(k[0]), state.board.m)
            b.minefield = m[0]
            b.knowledge = k[0]
            s = State(b)
            c = symm_coord(cell[0], cell[1], state.board.knowledge, nr=k[1], flip=k[2])
            self.__mono_update(s, c)

    def __mono_update(self, state, cell):
        o = Observation(state.board.knowledge, state.board.m)
        actions = self.P.get(o, dict())
        occurence = actions.get(cell, 0)
        occurence += 1
        self.P.update({o:{cell:occurence}})
        #self.P[o][cell] = occurence


    def update(self, state, cell):
        if self.sym:
            self.__sym_update(state,cell)
        else:
            self.__mono_update(state, cell)


    def train(self, board):
        def game_over(val, board):
            return val == MINE or board.win()

        assert isinstance(board, Board)
        state = State(board)
        # first move on a corner
        corners = [(0, 0), (0, board.w-1), (board.h-1, 0), (board.h-1, board.w-1)]
        r,c = random.choice(corners)
        val = state.probe(r,c, log=False)
        
        #board.draw(board.knowledge)
        last = False
        # main loop
        while not game_over(val, board) or not last:
            valid = []
            # fill array with cells on fringe that are not mines
            for rm,cm in state.fringe:
                if state.board.minefield[rm][cm] != MINE:
                    valid.append((rm, cm))
            # update P value of all valid actions
            for cell in valid:
                self.update(state, cell)
            if game_over(val, board):
                last = True
                continue
            # probe random cell in valid fringe
            r,c = random.choice(valid) if len(valid) > 0 else random.choice(tuple(state.uncovs))
            val = state.probe(r, c, log=False)
            #board.draw(board.knowledge)

        save_obj(self.P, "P{}".format(self.ind))

    def best_action(self, a_v_map):
        random.shuffle(a_v_map)
        return max(a_v_map)

    def next_action(self, state): 
        if not self.sym:
            s = self.P.get(state, None )
            if s:
                return self.best_action(s)
            else:
                return random.choice(tuple(state.fringe.union(state.uncovs)))
        else:
            for k,m in zip(symmetries(state.board.knowledge), symmetries(state.board.minefield)):
                b = Board(len(k), len(k[0]), state.board.m)
                b.minefield = m[0]
                b.knowledge = k[0]
                s = State(b)
                action_pool =  self.P.get(s, None)
                if action_pool:
                    return self.best_action(action_pool)
        
            return random.choice(tuple(state.fringe.union(state.uncovs)))


def train_Qplayer(rounds, qplayer, h, w, m):
    assert isinstance(qplayer, QPlayer)
    for r in range(rounds):
        print(r)
        qplayer.train(Board(h,w,m))
