"""
Model of the Tiger problem.
State:
    the Tiger is either behind the left (0) or the right (1) door

Action:
    open left door
    open right door

Observation:
    hear the tiger from the left
    hear the tiger from the right

Reward:
    wrong door: -100
    correct door: +10
    listening: -1
"""
from mdp.pomdp import POMDPState, POMDPObservation, POMDPAction, DecisionProcess
import random

LEFT = 0
RIGHT = 1

class State(POMDPState):
    def __init__(self, direction):
        self.d = direction
        self.doors = (False, False)

    def __hash__(self):
        return self.d

    def __str__(self):
        s = ' tiger on the '
        s += "left" if LEFT else "right"
        s += ', doors: {}'.format(self.doors)
        return s

    def __eq__(self, o):
        return self.d == o.d

    def clone(self):
        return State(self.d)

    def is_goal(self):
        return self.doors[self.d]

class Observation(POMDPObservation):
    def __init__(self, direction=None):
        hear = [False, False]
        if direction is not None:
            hear[direction] = True
        self.hear = tuple(hear)
    
    def __eq__(self, o):
        return self.hear == o.hear

    def __str__(self):
        return str(self.hear)

    def __hash__(self):
        return hash(self.hear)
    
    def available_actions(self):
        yield Action(listen=True)
        for i in range(len(self.hear)):
            yield Action(i)
    
    def __is_pref_action(self, h, a):
        """
        Prefered actions: open the door from which emanes the noise
        """
        if len(h) == 0 or not isinstance(a, Action):
            return False
        elif not a.listen:
            try:
                return self.hear[a.d] 
            except AttributeError:
                return False

    def V_init(self, h, a):
        return 12 if self.__is_pref_action(h, a) else 0
    
    def N_init(self, h, a):
        return 5 if self.__is_pref_action(h, a) else 0
        
class Action(POMDPAction):
    def __init__(self, direction=None, listen=False):
        self.d = 0
        if direction is not None:
            self.d += direction
        self.listen = listen
    
    def __str__(self):
        if not self.listen:
            assert isinstance(self.d, int)
            direction = "left" if self.d == LEFT else "right"
            return "open {} door".format(direction)
        else:
            return "listening"
    
    def __eq__(self,o):
        if not isinstance(o, Action):
            return False 
        return self.listen == o.listen and self.d == o.d
    
    def __hash__(self):
        if self.listen:
            return hash(self.listen)
        else:
            return hash(self.d)

    def do_on(self, state):
        assert isinstance(state, State)
        r = 0
        if self.listen:
            r = -1
        else:
            r = 10 if self.d == state.d else -100
            state = State(self.d)
        return (Observation(state.d), r)
        
class Tiger(DecisionProcess):
    def initial_belief(self):
        return random.choice([State(i) for i in range(2)])
        