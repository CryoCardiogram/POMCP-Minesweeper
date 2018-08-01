from abc import ABCMeta, abstractmethod

class POMDPObservation(metaclass=ABCMeta):
    """
    Abstract class for an Observation of the POMDP model.

    This class has to be extended regarding the application.
    """
    @abstractmethod
    def available_actions(self):
        """
        Generates all valid actions from the current observation

        Yields:
            POMDPAction: the next available action for the current observation
        """
        pass

    @abstractmethod
    def __eq__(self, oth):
        pass
    
    @abstractmethod
    def __hash__(self):
        pass

class POMDPState(metaclass=ABCMeta):
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

    @abstractmethod
    def is_goal(self):
        """
        Test if it is a goal state
        """
        pass

class POMDPAction(metaclass=ABCMeta):
    """
    Abstract class for an Action of the POMDP model. 
    This class Has to be extended depending on the application.
    """
    @abstractmethod
    def __eq__(self, oth):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def do_on(self, state):
        """
        Transition function. Perform the current action on the given state.
        
        Args:
            state (POMDPState): valid state on which to perform the action

        Return:
            POMDPObservation: the observation resulting from performing the action. 
            This methods modifies the state variable.
        """
        pass
    

        