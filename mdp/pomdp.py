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
    

        