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
        return self.actions == other.actions and self.observs == other.observs
        
    def __str__(self):
        s = ''
        for i in range(len(self)):
            s += '\n[{}] a: {} o:{}\n'.format(i, self.actions[i], self.observs[i])
        return s


    def __len__(self):
        return len(self.actions)

    def clone (self):
        """
        Return:
            History: clone of the current history
        """
        h = History()
        for a,o in zip(self.actions, self.observs):
            h.add(a,o)
        return h

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