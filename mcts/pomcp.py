from mdp.pomdp import POMDPAction, POMDPObservation, POMDPState
from mdp.history import History
from mcts.tree import Node
import scipy.signal as signal
import math
import random

params = {
    'K': 50,            # number of particles (size of the belief state space)
    'c': 0,             # exploration / exploitation ratio scalar constant (domain specific)
    'epsilon': 0.0,     # history discount factor
    'gamma': 1,         # reward discount factor
    'R_lo': 0,          # lowest value V(h) reached 
    'R_hi': 1           # highest value V(h) reached
}

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
    l = [ (child.a, UCB1(child, node.N)) for child in node.children ]

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

def simulate(state, node, depth):
    """
    This function implement the expansion and the backpropagation phase of a MCTS.
    During the expansion phase, if it is not the end, new nodes are creates from available actions.
    Information about the playouts (in rollout) are then updated during the backpropagation phase.
    """
    assert isinstance(node, Node)
    if end_rollout(depth, node.h):
        return 0
    
    if not node.is_intree(node.h):
        node.create_children()

def search(node):
    pass

