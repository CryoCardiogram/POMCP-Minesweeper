from mdp.pomdp import POMDPAction, POMDPObservation, POMDPState
from mcts.tree import Node
import math

params = {
    'K': 50,            # number of particles (size of the belief state space)
    'c': 0,             # exploration / exploitation ratio scalar constant (domain specific)
    'epsilon': 0.0,     # horizon discount factor
    'gamma': 1          # reward discount factor
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

