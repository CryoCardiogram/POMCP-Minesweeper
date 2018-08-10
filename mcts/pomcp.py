from mdp.pomdp import POMDPAction, POMDPObservation, POMDPState, DecisionProcess
from mdp.history import History
from mcts.tree import Node, create_node
import scipy.signal as signal
import math
import random
import time


params = {
    'K': 50,            # number of particles (size of the belief state space)
    'c': 0,             # exploration / exploitation ratio scalar constant (domain specific)
    'epsilon': 0.0,     # history discount factor
    'gamma': 1,         # reward discount factor
    'R_lo': 0,          # lowest value V(h) reached 
    'R_hi': 1,          # highest value V(h) reached
    'timeout':120,      # timeout for each iteration in seconds
    'start_time': 0,    # start time in seconds
    'max_depth': 20,    # max depth
    'root': Node(POMDPAction(), History(), 0, 0, list())
}


# start time
start_time = 0


def UCB1_action_selection(node, greedy=False):
    """
    Implementation of the UCB1 algorithm for solving a multi-armed bandit problem.

    https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf

    Each action a available from the history h are assigned a value V(ha), 
    computed from simulations of the POMDP from the history h.
    In non-greedy mode, this value is augmented by an exploration bonus for rarely-tried actions.

    Args:
        node (Node): root node of the tree, containing history h
        greedy (bool): enable/disable greedy mode

    Return:
        (POMDPAction, float): best action and its UCB1 value 
    """
    assert isinstance(node, Node)
    assert node.inTree
        
    def UCB1(child, N):
        """
        UCB1 formula, return infinity if N == 0
        """
        assert isinstance(child, Node)
        v = child.V
        if greedy:
            return v
        try:
            v+= params['c']*math.sqrt(math.log(N)  /child.N) 
        except ZeroDivisionError:
            v = math.inf
        except ValueError:
            v = math.inf
        finally:
            return v
             
    # (action, UCB1val) list 
    l = [ (a, UCB1(child, node.N)) for a, child in node.children.items() ]
    if greedy:
        print("tree history {}".format(node.h))
        print( [(a, (child.N, child.V)) for a, child in node.children.items() ]  )
    return max(l, key=lambda t: t[1])

def discount_calc(rewards, discount):
    """
    Vectorized discount computation.

    C[i] = R[i] + discount * C[i+1]
    signal.lfilter(b, a, x, axis=-1, zi=None)
    a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
                          - a[1]*y[n-1] - ... - a[N]*y[n-N]
    """
    r = rewards[::-1]
    a = [1, -discount]
    b = [1]
    y = signal.lfilter(b, a, x=r)
    return y[::-1]

def end_rollout(depth, h):
    """
    Predicate to test simulations ending criterion 
    """
    assert isinstance(h, History)
    if params['gamma']**depth < params['epsilon'] or depth >= params['max_depth']:
        #print("max depth")
        return True
    elif h.last_obs().is_terminal():
        #print("terminal obs")
        return True
    elif (time.time() - params['start_time']) >= params['timeout']:
        #print("time out")
        #print(time.time() - params['start_time'])
        return True
    else:
        return False

def rollout(state, node, depth):
    """
    This function simulates the process with a random action policy, 
    from the given start state until a depth threshold is met (controlled by the epsilon param)
    Args:
        state (POMDPState): the start state
        node (Node): node with current history h
        depth (int): current depth in the tree
    Return:
        float: the final reward of the random playout 
    """
    assert isinstance(node, Node)
    assert isinstance(state, POMDPState)

    s = state.clone()
    h = node.h.clone()
    d = depth
    rewards = []
    while not end_rollout(d, h):
        # iterative implementation
        a = random.choice([action for action in h.last_obs().available_actions()])
        o, r = a.do_on(s)
        rewards.append(float(r))
        d += 1
        h.add(a, o)
    #print(rewards)
    return discount_calc(rewards, params['gamma'])[0] if len(rewards) > 0 else 0

def simulate(state, node, proc=None):
    """
    Iterative implementation of an MCTS simulation step, adapted to partial observability. This function builds 
    a whole PO-MCTS starting from the root node, alternating between the following phases.

    Expansion: if the termination criterion is not met, new nodes are created from  currently available actions.

    Selection: select the best node among the children evaluated with their UCB1 value.

    Simulation: simulate a playout starting from the selected node 

    Backpropagation: update statistics about the playouts (in rollout) up to the root.

    Args:
        state (POMDPState): state sampled either from the initial state distribution or from the belief space
        node (Node): current root of the tree containing the current history
        proc (DecisionProcess): domain specific knowledge about the pomdp
    
    """
    assert isinstance(node, Node)
    depth = 0
    rewards = []
    root = node
    fringe = [(node, depth)] # descending down the tree
    backprop = [] # climbing up the tree
    s = state.clone()
    max_d = 0
    while fringe:
        nod, d = fringe.pop()
        

        if end_rollout(d, nod.h):
            rewards.append(0)
            backprop.append((nod, d, s.clone()))
            continue

        max_d = d if d >= max_d else max_d

        if not root.is_intree(nod.h):
            # Expansion
            #print("expansion d:{} a: {}".format(d, nod.a))
            #for a, c in root.children.items():
            #    print("child {}".format(a))
            #    print(c.h)
            #    print(c.h == nod.h and c.inTree)
            nod.create_children()
            nod.inTree = True
            backprop.append((nod, d, s.clone()))
            rewards.append(rollout(s, nod, d))
            continue
        backprop.append((nod, d, s.clone()))

        # Selection
        a,u = UCB1_action_selection(nod)
        #print("selection d:{} a: {}".format(d, a))
        # Simulation
        o, r = a.do_on(s)
        #hao = nod.h.clone()
        #nod.children[a] = create_node(hao, a, o)
        nod.children[a].h.add(a,o)
        #nod.children[a].inTree = True
        rewards.append(float(r))
        fringe.append((nod.children[a], d+1))
    
    # Backpropagation
    for i in range(1, len(backprop)+1):
        nod, d, s = backprop[-i] # parent
        nod_a = backprop[-i + 1][0] # simulated child 
        R = discount_calc(rewards[d::], params['gamma'])[0]
        nod.B.append(s)
        #print("d{}, h:{}, bsize: {}".format(d, nod.h, len(node.B)))
        nod_a.N += 1
        nod_a.V += (R - nod_a.V) / nod_a.N 
    
    #print("root belief size: {}".format(len(root.B)))
    #print("max depth:{}".format(max_d))
    # particles invigoration
    if proc:
        assert isinstance(proc, DecisionProcess)
        for a, child in node.children.items():
            #if len(child.B):
                #print("child {}, bsize: {}".format(a, len(child.B)))
            proc.invigoration(child.B)

def search(h, proc, max_iter, clean=False):
    """
    This function implements the UCT algorithm.

    Args:
        h (History): history in the current root of the tree
        proc (DecisionProcess): model of domain knowledge of the pomdp
        max_iter (int): maxium number of iterations
        clean (bool): toggle to reset the tree

    Return:
        POMDPAction: the optimal action
    """
    assert isinstance(h, History)
    assert isinstance(proc, DecisionProcess)
    # init global vars
    params['start_time'] = time.time()
    if clean:
        params['root'] = Node(POMDPAction(), History(), 0, 0, list())
    root = params['root']

    # define the new root node
    print("current root: {}, len(h): {}".format(h.actions[0], len(h))) 
    treeroot = Node(h.last_action(), h, 0, 0, list())   
    ite = 0
    # time out
    def time_remaining():
        return ite < max_iter and (time.time() - params['start_time']) < params['timeout']
    
    while time_remaining():
        s = proc.initial_belief()
        if len(treeroot.h) > 1:
            s = random.choice(root.B)
        simulate(s, treeroot , proc)
        ite+=1   
    # greedy action selection
    a = UCB1_action_selection(treeroot, greedy=True)[0]
    params['root'] = treeroot
    print("next belief size: {}".format(len(treeroot.children[a].B)))
    return a
    
    