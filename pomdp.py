class POMDPObservation(object):
    """
    Abstract class for an Observation of the POMDP model.

    This class has to be extended regarding the application.
    """

    def available_actions(self):
        """
        Generates all valid actions from the current observation

        Yields:
            POMDPAction: the next available action for the current observation
        """
        pass
    
    def __eq__(self, oth):
        pass
    
    def __hash__(self):
        pass

class POMDPState(object):
    """
    Abstract class for a State of the POMDP model. 
    This class Has to be extended depending on the application.
    """
    def successors(self, action):
        """
        Generates all observations resulting from performing the given action 
        on the current state. 

        Args:
            action (POMDPAction): valid action done on the current state

        Yields: 
            POMDPObservation: the next possible observation resulting from doing the action
            on the current state.
        """
        pass

class POMDPAction(object):
    """
    Abstract class for an Action of the POMDP model. 
    This class Has to be extended depending on the application.
    """

    def __eq__(self, oth):
        pass
    


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
        self.actions.insert(0, action)
        self.observs.insert(0, observation)
    
    def last_action(self):
        return self.actions[0]
    
    def last_obs(self):
        return self.observs[0]
        