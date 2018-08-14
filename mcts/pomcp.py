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
    'log': 1,           # level of logs printed on console [0,2]
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
    assert node.inTree, "{} child(ren), {}".format(len(node.children), node.h)

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
    a,f = max(l, key=lambda t: t[1])
    if greedy and params['log'] >= 2:
        print("tree history {}".format(node.h))
        print( [(a, (child.N, child.V)) for a, child in node.children.items() ]  )
    return (a,f)

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
    elif len(h) > 0 and h.last_obs().is_terminal():
        #print("terminal obs")
        return True
    elif (time.time() - params['start_time']) >= params['timeout']:
        #print("time out")
        #print(time.time() - params['start_time'])
        return True
    else:
        return False

def rollout(state, node, depth, policy=None):
    """
    This function simulates the process with a defined action policy (random by default), 
    from the given start state until a depth threshold is met (controlled by the epsilon param)
    Args:
        state (POMDPState): the start state
        node (Node): node with current history h
        depth (int): current depth in the tree
        policy (History -> POMDPAction): function that takes a History as argument and return 
        a POMDPAction
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
        if policy:
            a = policy(h)
        else:
            action_pool = [action for action in h.last_obs().available_actions(h)]
            a = random.choice(action_pool)
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
        hao = nod.h.clone()
        #hao.add(a, o)
        #nod.children[a] = create_node(hao, a, o)
        #nod.children[a].h.add(a,o)
        #nod.children[a].inTree = True
        rewards.append(float(r))
        if nod.children[a].inTree:
            fringe.append((nod.children[a], d+1))
        else:
            nod.children[a] = create_node(hao, a, o)
            fringe.append((nod.children[a], d+1))
    
    # Backpropagation
    for i in range(1, len(backprop)+1):
        nod, d, s = backprop[-i] # parent
        nod_a = backprop[-i + 1][0] # simulated child 
        R = discount_calc(rewards[d::], params['gamma'])[0]
        # only add s to the belief space if its observation match the real observation
        sc = state.clone()
        o = nod.h.last_obs()
        for i in range(1, d):
            act = nod.h.actions[i]
            o, tmp = act.do_on(sc)
        if nod.h.last_obs() == o:
            nod.B.append(s)
            #if d >= 1:
                #print("oyo")
        #else:
        #    print("wrong belief update?")
        #print("d{}, h:{}, bsize: {}".format(d, nod.h, len(node.B)))
        nod_a.N += 1
        nod_a.V += (R - nod_a.V) / nod_a.N 
    
    #print("root belief size: {}".format(len(root.B)))
    #print("max depth:{}".format(max_d))
    

def search(h, proc, max_iter, clean=True):
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
        params['root'] = Node(h.last_action(), h, 0, 0, list())
    
    root = params['root'].children[h.last_action()] if h.last_action() != POMDPAction() else params['root']

    # root should have history given as args but B from previous root
    root.h = h.clone()
    # at each call to search, children of the current root must be regenerated, to 
    # consider the last real action-observation obtained
    root.inTree = False

    if params['log'] >= 1:
        print("current root: {}, len(h): {}".format(h.actions[0], len(h)))    
    ite = 0
    # time out
    def time_remaining():
        return ite < max_iter and (time.time() - params['start_time']) < params['timeout']
    
    # search
    while time_remaining():
        s = proc.initial_belief()
        if len(root.h) > 1:
            s = random.choice(tuple(root.B))
        simulate(s, root , proc)
        ite+=1   

    updateR(root)

    # greedy action selection
    a = UCB1_action_selection(root, greedy=True)[0]
    params['root'] = root
    child = root.children[a]

    # particle reinvigoration
    proc.invigoration(child.B, ite)
    if params['log'] >= 1:
        print("next belief size: {}".format(len(child.B)))
    return a
    
def updateR(root):
    assert isinstance(root, Node)
    with open("hi_lo_R.txt", "r") as o:
        params['R_lo'], params['R_hi'] = tuple(float(l) for l in o.readlines())
    hi = -math.inf
    lo = math.inf
    for a, child in root.children.items():
        if child.V > hi: 
            hi = child.V
        if child.V < lo:
            lo = child.V
    params['R_lo'] = lo if lo < params['R_lo'] else params['R_lo']
    params['R_hi'] = hi if hi > params['R_hi'] else params['R_hi']
    with open("hi_lo_R.txt", "w") as o:
        o.write("{}\n{}".format(params['R_lo'], params['R_hi']))
