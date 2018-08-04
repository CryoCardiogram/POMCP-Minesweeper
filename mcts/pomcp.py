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
    'timeout_s':120     # timeout for each iteration in seconds
}

# current root of the node tree
root = Node(POMDPAction(), History(), 0, 0, list())

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
            v+= params['c']*math.sqrt( math.log(child.N) / N) 
        except ZeroDivisionError:
            v = math.inf
        finally:
            return v
             
    # (action, UCB1val) list 
    l = [ (a, UCB1(child, node.N)) for a, child in node.children.items() ]

    return max(l, key=lambda t: t[1])

def discount_calc(rewards, discount):
    """
    Vectorized discount computation.

    C[i] = R[i] + discount * C[i+1]
    signal.lfilter(b, a, x, axis=-1, zi=None)
    a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
                          - a[1]*y[n-1] - ... - a[N]*y[n-N]
    """
    a = [1, -discount]
    b = [1]
    y = signal.lfilter(b, a, x=rewards)
    return y[::-1]

def end_rollout(depth, h):
    """
    Predicate to test simulations ending criterion 
    """
    assert isinstance(h, History)
    if params['gamma']**depth < params['epsilon']:
        return True
    elif h.last_obs().is_terminal():
        return True
    elif (time.time() - start_time) >= params['timeout']:
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

    return discount_calc(rewards, params['gamma'])[0]

def simulate(state, node, proc=None):
    """
    Iterative implementations of simulative part of an MCTS, adapted to partial observability. This function builds 
    a whole PO-MCTS starting from the root node.

    Expansion: if it is not the end, new nodes are creates from available actions.

    Selection: a node is selected among the children, evaluated with the UCB1 function.

    Simulation: a playout is simulated starting from the selected node 

    Backpropagation: Information about the playouts (in rollout) are then updated during this phase.

    Args:
        state (POMDPState): state sampled either from the initial state distribution or from the belief space
        node (Node): current root of the tree containing the current history
        proc (DecisionProcess): domain specific knowledge about the pomdp
    
    """
    assert isinstance(node, Node)
    depth = 0
    rewards = []

    fringe = [(node, depth)] # descending down the tree
    backprop = [] # climbing up the tree
    s = state.clone()
    while fringe:
        nod, d = fringe.pop()

        if end_rollout(depth, nod.h):
            rewards.append(0)
            backprop.append((nod, d, s.clone()))
            continue

        if not root.is_intree(nod.h):
            # Expansion
            nod.create_children()
            nod.inTree = True
            backprop.append((nod, d, s.clone()))
            rewards.append(rollout(s, nod, d))
            continue
        
        backprop.append((nod, d, s.clone()))

        # Selection
        a = UCB1_action_selection(node)[0]
    
        # Simulation
        o, r = a.do_on(s)
        hao = nod.h.clone().add(a,o)
        node_hao = create_node(hao, a, o)
        rewards.append(float(r))
        fringe.append((node_hao, d+1))

    # Backpropagation
    for i in range(2, len(backprop)-1, 1):
        nod, d, s = backprop[-i] # parent
        nod_a = backprop[-i + 1][0] # simulated child 
        R = discount_calc(rewards[d::], params['gamma'])[0]
        nod.B.append(s)
        nod.N += 1
        nod_a.N += 1
        nod_a.V += (R - nod_a.V) / nod_a.N 
    
    # particles invigoration
    if proc:
        assert isinstance(proc, DecisionProcess)
        for a, child in node.children.items():
            proc.invigoration(child.B)

def search(node, proc, max_iter):
    """
    This function implements the UCT algorithm.

    Args:
        node (Node): current root of the tree
        proc (DecisionProcess): model of domain knowledge of the pomdp
        max_iter (int): maxium number of iterations

    Return:
        POMDPAction: the optimal action
    """
    assert isinstance(node, Node)
    assert isinstance(proc, DecisionProcess)
    # init global vars
    start_time = time.time()
    root = node 
    ite = 0
    # time out
    time_remaining =  ite < max_iter and (time.time() - start_time) < params['timeout']
    while time_remaining:
        s = proc.initial_belief()
        if len(root.h) != 0:
            s = random.choice(root.B)
        simulate(s,node , proc)
    # greedy action selection
    return UCB1_action_selection(root, greedy=True)[0]
        

