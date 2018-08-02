from mdp.pomdp import POMDPAction, POMDPObservation
from mdp.history import History
import numpy as np

def n_init(h, a):
    """
    Args:
        h (History): previous history
        a (POMDPAction): next action
    
    Returns: 
        int: initial N(h,a) value
    """
    o = h.last_obs()
    return o.N_init(h,a)

def v_init(h,a):
    """
    Args:
        h (History): previous history
        a (POMDPAction): next action
    
    Returns: 
        int: initial Q(h,a) value
    """
    o = h.last_obs()
    return o.V_init(h,a)


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
        B (set): collection of K particles (states), representing the current belief of the system
        children (list): list of child-node, sorted by actions
        inTree (bool): set to True if the history of the node is up to date 
    """
    def __init__(self, a, h, V, N, B):
        assert isinstance(h, History)
        assert isinstance(a, POMDPAction)
        self.h = h 
        self.a = a
        self.V = V 
        self.N = N 
        self.B = B
        self.children = dict()
        self.inTree = False

    def create_children(self):
        """
        Inizialize children nodes with respect to available actions
        for the current history. 

        """
        o = self.h.last_obs()
        for a in o.available_actions():
            # updated history 
            ha = self.h.clone()
            self.children.update(  {a: Node(a, ha, v_init(ha, a), n_init(ha, a), list() )} )
        
    def add_inTree(self, obs):
        """
        Add the current node to the Tree
        """
        assert isinstance(obs, POMDPObservation)
        self.h.add(self.a, obs)
        self.inTree = True

    def get_child(self, a):
        """
        Return:
            Node: the child node associated with the action a, or None.
        """ 
        self.children.get(a)

    def is_intree(self, h):
        """
        Search the history h in the tree. 
        Explore the tree in BFS and stop the search as soon as the size of histories 
        in the tree become larger than len(h)

        Args:
            h (History): history to look for. len(h) is greater or equals than 
            the history of the root. 
        """
        assert isinstance(h, History)
        fringe = [self]
        while fringe:
            node = fringe.pop()
            if len(node.h.actions) <= len(h.actions):
                if node.h == h:
                    return True
                else :
                    for a in node.children:
                        fringe.append(node.children[a])
            else:
                return False





