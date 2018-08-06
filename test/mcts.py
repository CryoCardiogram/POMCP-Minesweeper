import unittest
import math
import time
from timeit import Timer
from mcts.tree import Node, create_node
from mdp.history import History
from mdp.pomdp import POMDPAction, POMDPObservation, POMDPState, DecisionProcess
from problems.tiger.model import State, Action, Observation, Tiger, LEFT, RIGHT
from mcts.pomcp import (UCB1_action_selection, discount_calc, end_rollout, rollout, 
    params, simulate, search)

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
        self.assertEqual(12, node.V)
        self.assertEqual(5, node.N)

class TestPOMCP(unittest.TestCase):
    def setUp(self):
        self.pomdp = Tiger()
        self.start = State(LEFT) # tiger behind left door
        o = Observation()
        h = History()
        a = POMDPAction()
        self.root = create_node(h, a, o)
        self.root.inTree = True
        
        params['c'] = 2
        # setup a tree with root and its depth-1 children
        self.root.create_children()
        # expand listen-node
        a = Action(listen=True)
        s = self.start.clone()
        o, r = a.do_on(s)
        self.listen_child = self.root.children[a]
        self.listen_child.h.add(a, o)
        self.listen_child.inTree = True
        self.listen_child.create_children()
        # There is 3 depth-2 nodes, their histories go like: 
        # empty-listen-listen (ell), empty-listen-left (elf), empty-listen-right (elr)
        # elr should have highest V (pref action)
        for act, child in self.listen_child.children.items():
            obs, r = child.a.do_on(s.clone())
            self.listen_child.children[act] = create_node(child.h, act, obs)
            self.listen_child.children[act].inTree = True
        
    def test_ucb1_sel(self):
        a, v = UCB1_action_selection(self.root)
        self.assertEqual(0, self.root.N)
        # infinity and lexicographical
        self.assertEqual(Action(listen=True), a)
        self.assertEqual(math.inf, v)
        self.root.N = 1
        # greedy
        self.listen_child.N = 2
        a, v = UCB1_action_selection(self.listen_child, greedy=True)
       
        self.assertEqual(Action(direction=LEFT), a)
        # default
        for a, c in self.listen_child.children.items():
            c.N += 1
        a, v = UCB1_action_selection(self.listen_child)
        self.assertEqual(Action(direction=LEFT), a)
        self.assertEqual(self.listen_child.children[a].V + params['c']*math.sqrt(math.log(self.listen_child.N) / self.listen_child.children[a].N), v)
        
    def test_discount_calc(self):
        r = [ 1, 2, 1]
        dis = 0.9
        self.assertEqual(discount_calc(r, dis)[0], r[0] + dis * (r[1] + dis* (r[2]) ))

    def test_end_rollout(self):
        params['gamma'] = 0.9
        params['epsilon'] = 0.9
        params['start_time'] = time.time()
        r = rollout(self.start, self.root, 0)
        #print(r)
        self.assertIsNotNone(r)

        params['epsilon'] = 0.000000001 # many  steps
        params['gamma'] = 0.99999
        params['timeout'] = 3  

        # test timeout
        def procedure():
            params['start_time'] = time.time()
            r = rollout(self.start, self.root, 0) 

        t = Timer(stmt=procedure)   
        self.assertAlmostEqual(t.timeit(1), params['timeout'], delta=1)
