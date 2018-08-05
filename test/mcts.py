import unittest
from mcts.tree import Node, create_node
from mdp.history import History
from mdp.pomdp import POMDPAction, POMDPObservation, POMDPState, DecisionProcess
from problems.tiger.model import State, Action, Observation, Tiger, LEFT, RIGHT

class TestTree(unittest.TestCase):
    def setUp(self):
        self.pomdp = Tiger()
        self.start = State(LEFT) # tiger behind left door
        self.o = Observation()
        self.h = History()
        self.a = POMDPAction()

    def test_create_node(self):
        node = create_node(self.h, self.a, self.o)
        self.assertTrue(isinstance(node, Node))
        self.assertEqual(self.a, node.a)
        self.assertEqual(len(self.h), 1)
        self.assertFalse(node.inTree)
    
    def test_create_children(self):
        node = create_node(self.h, self.a, self.o)
        node.create_children()
        self.assertEqual(3, len(node.children))
        for act, child in node.children.items():
            self.assertFalse(child.inTree)
    
    def test_is_in_tree(self):
        # setup a tree with root and its depth 1 children
        root = create_node(self.h, self.a, self.o)
        root.inTree = True
        root.create_children()
        for act, child in root.children.items():
            obs, r = child.a.do_on(self.start.clone())
            child.h.add(child.a, obs)
            child.inTree = True

        h = root.h.clone()
        a = Action(listen=True)
        s = self.start.clone()
        o, r = a.do_on(s)
        h.add(a, o)
        self.assertTrue(root.is_intree(h))
        h2 = h.clone()
        h2.add(a, o)
        self.assertFalse(root.is_intree(h2))

    
    def test_pref_actions(self):
        self.h.add(self.a, self.o)
        a = Action(listen=True)
        s = self.start.clone()
        o, r = a.do_on(s)
        self.h.add(a, o)
        a2 =  Action(direction=LEFT)
        o2, r = a2.do_on(s)
        node = create_node(self.h, a2, o2)
        self.assertEqual(100, node.V)
        self.assertEqual(5, node.N)
