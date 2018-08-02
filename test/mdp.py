import unittest
from minesweeper.board import Board
from minesweeper.globals import UNCOV, ONE
from minesweeper.state import State, Action, Observation
from mdp.pomdp import POMDPAction, POMDPObservation, POMDPState
from mdp.history import History

class TestPOMDP(unittest.TestCase):
    def setUp(self):
        self.b = Board(2, 2, 1)
        self.s = State(self.b)
        self.b2 = Board(3, 3, 8)
        self.s2 = State(self.b2)

    def test_do_on(self):
        a = Action(0, 0)
        o = a.do_on(self.s)
        self.assertEqual( o.K, [[ONE, UNCOV], [UNCOV, UNCOV]] )
        self.assertEqual(o.K, self.s.board.knowledge)

    def test_is_goal(self):
        a = Action(1, 1)
        o = a.do_on(self.s2)
        self.assertTrue(self.s2.is_goal())

    def test_available_actions(self):
        a = Action(1, 1)
        o = a.do_on(self.s)
        l = []
        for a in o.available_actions():
            l.append(a)
        self.assertIn(Action(0, 0), l)
        self.assertIn(Action(1, 0), l)
        self.assertIn(Action(0, 1), l)

class TestHistory(unittest.TestCase):
    def setUp(self):
        self.b = Board(4, 5, 3)
        self.s = State(self.b)
        self.h = History()

    def test_add(self):
        a = Action(0, 0)
        o = a.do_on(self.s)
        self.h.add(a, o)
        a2 = Action(2, 1)
        o2 = a2.do_on(self.s)
        self.h.add(a2, o2)
        self.assertEqual(self.h.last_action(), a2)
        #print(o2)
        #print(self.h.last_obs())
        self.assertEqual(self.h.last_obs(), o2)
        
    
    def test_clone(self):
        a = Action(1, 0)
        o = a.do_on(self.s)
        self.h.add(a, o)
        h  = self.h.clone()
        self.assertEqual(h, self.h)
        
    