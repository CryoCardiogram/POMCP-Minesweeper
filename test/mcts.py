import unittest
from mcts.tree import Node, create_node
from mdp.history import History
from mdp.pomdp import POMDPAction, POMDPObservation, POMDPState, DecisionProcess
from problems.tiger.model import State, Action, Observation, Tiger

class TestTree(unittest.TestCase):
    def setUp(self):
        self.pomdp = Tiger()
        self.start = self.pomdp.initial_belief()

    def test_create_node(self):
        o = Observation()
        h = History()
        a = POMDPAction()
        node = create_node(h, a, o)
        self.assertTrue(isinstance(node, Node))
        self.assertEqual(a, node.a)
        self.assertEqual(len(h), 1)

