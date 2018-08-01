from mdp.pomdp import POMDPAction, POMDPObservation

class History(object):
    """

    """
    def __init__(self):
        self.actions = list()
        self.observs = list()
    
    def __hash__(self):
        t = tuple([(a,o) for a,o in zip(self.actions, self.observs)])
        return hash(t)

    def __eq__(self, other):
        for a,o in zip(other.actions, other.observs):
            if self.actions != other.actions:
                return False
            elif self.observs != other.observs:
                return False
        return True

    def clone (self,h):
        """
        Erase the history of the current object to clone the history of h
        """
        assert isinstance(h, History)
        self.actions.clear
        self.observs.clear
        for a,o in zip(h.actions, h.observs):
            self.add(a,o)

    def add(self, action, observation):
        """
        Update the history by adding a new action-observation pair.
        """
        assert isinstance(action, POMDPAction)
        assert isinstance(observation, POMDPObservation)
        self.actions.insert(0, action)
        self.observs.insert(0, observation)
    
    def last_action(self):
        a = self.actions[0]
        assert isinstance(a, POMDPAction)
        return a
    
    def last_obs(self):
        o = self.observs[0]
        assert isinstance(o, POMDPObservation)
        return o