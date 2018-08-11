from mdp.pomdp import POMDPAction, POMDPObservation
from mdp.history import History
import numpy as np

def n_init(h, a):
    """
    Args:
        h (History): previous history
        a (POMDPAction): next action (not in history)
    
    Returns: 
        int: initial N(h,a) value
    """
    o = h.last_obs()
    return o.N_init(h,a)

def v_init(h,a):
    """
    Args:
        h (History): previous history
        a (POMDPAction): next action (not in history)
    
    Returns: 
        int: initial Q(h,a) value
    """
    o = h.last_obs()
    return o.V_init(h,a)

def create_node(h, a, o):
    """
    Args: 
        h (History): history prior to the node
        a (POMDPAction): next action (not in history)
        o (POMDPObservation): next observation (not in history)

    Return:
        Node: a new tree node whose attributes value comes from domain knowledge
    """
    assert isinstance(h, History)
    h.add(a, o)
    n = Node(a, h, v_init(h,a), n_init(h,a), list())
    #n.inTree = True
    return n

class Belief(set):
    def append(self, a):
        # for retrocompatibility
        self.add(a)

class Node(object):
    """
    Each node T(h) is defined by the tuple <N(h), V(h), B(h)>
    inTree attribute is set to False when the node is initially constructed, 
    as its history does not contain its last action-observation yet. 
    
    Attributes: 
        a (POMDPAction): action performed to reach the node
        h (History): history to reach the node
        N (int): number of visits 
        V (float): estimation of the Q(h,a) value of the node
        B (list): collection of K particles (states), representing the current belief of the system
        children (dict): collection of child-node, sorted by actions
        inTree (bool): set to True if the history of the node is up to date 
    """
    def __init__(self, a, h, V, N, B):
        assert isinstance(h, History)
        assert isinstance(a, POMDPAction)
        self.h = h 
        self.a = a
        self.V = V 
        self.N = N 
        self.B = Belief()
        for e in B:
            self.B.add(e)
        self.children = dict()
        self.inTree = False

    def create_children(self):
        """
        Initialize children nodes with respect to available actions
        for the current history. 
        """
        o = self.h.observs[0]
        assert(str(self.h.actions[-1]) == '(empty)')
        for a in o.available_actions(self.h):
            # updated history 
            ha = self.h.clone()
            self.children.update(  {a: Node(a, ha, v_init(ha, a), n_init(ha, a), list() )} )
        
    def find(self, h):
        """
        Explore the tree in BFS and stop the search as soon as the size of histories 
        in the tree become larger than len(h)
        """     
        assert isinstance(h, History)
        fringe = [self]
        while fringe:
            node = fringe.pop()
            if len(node.h) < len(h):
                for a in node.children:
                    fringe.insert(0, node.children[a])
            elif len(node.h) > len(h):
                return None
            else:
                if node.h == h and node.inTree:
                    return node

    def is_intree(self, h):
        """
        Search the history h in the tree. 
    
        Args:
            h (History): history to look for. len(h) is greater or equals than 
            the history of the root. 
        
        Return:
            bool: whether there is node containing history h in the tree
        """
        if self.find(h):
            return True 
        return False
        
        