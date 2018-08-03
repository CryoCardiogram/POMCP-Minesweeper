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
    
    def V_init(self, h, a):
        """
        Q(h, a) initial value for this observation. Can override this function
        to initialize new nodes in the pomcp tree with specific values
        from domain knowledge of the pomdp.

        Args:
            h (History): previous history for the current observation
            a (POMDPAction): action resulting in the current observation
        """
        #TODO handle prefered actions
        return 0
    
    def N_init(self, h, a):
        """
        Initial number of visits N(h,a) for the History h + action a and observation o, the current observation. 
        Can override this function to initialize new nodes in the pomcp tree with specific values
        from domain knowledge of the pomdp.

        Args:
            h (History): previous history for the current observation
            a (POMDPAction): action resulting in the current observation
        """
        return 0
    
    @abstractmethod
    def is_terminal(self):
        """
        Return:
            bool: True if there is no available actions for this history
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
    @abstractmethod
    def clone(self):
        pass
        
    @abstractmethod
    def is_goal(self):
        """
        Test if it is a goal state
        """
        pass

class POMDPAction(object):
    """
    Base class for an Action of the POMDP model. An agent can generate empty 
    actions by creating instances of the class.
    This class has to be extended depending on the application.
    """
    
    def __init__(self, *args, **kwargs):
        self.__empty = False
        if len(args) == 0 and len(kwargs) == 0:
            self.__empty = True
        else :
            raise TypeError
            
    def __eq__(self, oth):
        """
        method to override
        """
        return isinstance(oth, POMDPAction) and oth.__empty 

    def __str__(self):
        """
        method to override
        """
        return "(empty)"

    def do_on(self, state):
        """
        Transition function that performs the current action on the given state. It is used 
        as a black-box simulator, in order to update the value function without knowing about the model's dynamics.
        
        Args:
            state (POMDPState): valid state on which to perform the action

        Return:
            (POMDPObservation, int): the observation resulting from performing the action and the intermediate reward.
            This method modifies the state variable.
        """
        pass

class DecisionProcess(metaclass=ABCMeta):
    """
    This ABC should be extended to describe meta-parameters of the POMDP. 
    """
    def invigoration(self, B):
        """
        Particle invigoration method to fill in the belief space with additional states, in order to 
        avoid the degeneracy problem.

        As the belief space for an an history h is approximated by a set of particles 
        corresponding to a sample state, after each action-observation, the particles are updated by 
        a Monte-Carlo sampling. 

        This particle filter approeach may suffer from particle deprivation when the time step of the process
        increases. A way to avoid that is to introduce new particles by adding artifiial noise to the belief space, from 
        domain knowledge. 

        Args:
            B (list): belief space approximated by an unweighted bag of states (particles)  
        """
        pass
    
