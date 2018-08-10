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

    def test_rollout(self):
        params['gamma'] = 0.9
        params['epsilon'] = 0.9
        params['start_time'] = time.time()
        r = rollout(self.start, self.root, 0)
        #print(r)
        self.assertIsNotNone(r)

    def test_end_rollout(self):
        params['epsilon'] = 0.000000001 # many  steps
        params['gamma'] = 0.99999
        params['max_depth']= 1000000000
        params['timeout'] = 3  

        # test timeout
        def procedure():
            params['start_time'] = time.time()
            r = rollout(self.start, self.root, 0) 

        t = Timer(stmt=procedure)   
        self.assertAlmostEqual(t.timeit(1), params['timeout'], delta=1)

    def test_simulate_expansion(self):
        root = create_node(History(), POMDPAction(), Observation())
        #self.root1 = root
        params.update({
            'start_time': time.time(),
            'gamma': 0.5,
            'epsilon': 0.2,
            'max_depth':100,
            'timeout': 3
        })
        simulate(self.start, root)
        self.assertEqual(len(root.children), 3)
        self.assertEqual(root.N, 1)

    def test_simulate_iteration_fulltree(self):
        self.root.N = 2
        self.root.V = 4/3
        self.listen_child.N = 2
        self.listen_child.V = 4
        for a, c in self.listen_child.children.items():
            c.N += 1
        # 1st iteration
        params.update({
            'start_time': time.time(),
            'gamma': 0.5,
            'epsilon': 0.26 ,    # depth 2 
            'timeout': 3,
            'max_depth': 100,
            'c': 2
        })
        simulate(self.start, self.root)
        ## left-door child (history: empty-left [gauche in french]) should have been chosen (exploration)
        eg = self.root.children[Action(LEFT)]
        self.assertEqual(len(eg.children), 3, msg="wrong child 1st iteration")
        self.assertGreater(eg.N, 0) # with prefered action, could be higher than 1
        # abs value should be high (could be negative)
        self.assertGreater(abs(eg.V), 5) 

        # 2nd iteration
        ## new sample from root belief space
        sr = State(RIGHT)
        params['start_time']: time.time()
        simulate(sr, self.root)
        ## right-door child chosen (exploration)
        er = self.root.children[Action(direction=RIGHT)]
        self.assertEqual(len(er.children), 3, msg='wrong child 2nd iteration')
        self.assertGreater(er.N, 0)
        # abs value should be high (could be negative)
        self.assertGreater(abs(er.V), 5)

        # if backpropagation works, root.N should be 4
        self.assertEqual(self.root.N, 4, msg='backpropagation fails')

        # 3rd iteration
        params['start_time']: time.time()
        sl = State(LEFT)
        simulate(sl, self.root)
        # simply test backpropagation for now
        self.assertEqual(self.root.N, 5)

    def test_search(self):
        self.root.N = 2
        self.root.V = 4/3
        self.listen_child.N = 2
        self.listen_child.V = 4
        for a, c in self.listen_child.children.items():
            c.N += 1
        params.update({
            'gamma': 0.5,
            'epsilon': 0.26 ,    # depth 2 
            'timeout': 5,
            'max_depth': 100,
            'c': 2
        })

        # timeout
        def proc():
            search(self.root.h,self.pomdp, 1000000)

        def proc2():
            params['timeout']=30
            search(self.root.h,self.pomdp, 10)

        t = Timer(stmt=proc)
        self.assertAlmostEqual(t.timeit(1), params['timeout'], delta=1)

        t = Timer(stmt=proc2)
        self.assertLessEqual(t.timeit(1), 100)

        a = search(self.root.h,self.pomdp, 100)
        self.assertTrue(isinstance(a, POMDPAction))

