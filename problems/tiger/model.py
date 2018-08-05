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

    def __eq__(self, o):
        return self.d == o.d

    def clone(self):
        return State(self.d)

    def is_goal(self):
        return self.doors[self.d]

class Observation(POMDPObservation):
    def __init__(self, direction=None):
        self.hear = [False, False]
        if direction:
            self.hear[direction] = True
        self.hear = tuple(self.hear)
    
    def __eq__(self, o):
        self.hear == o.hear

    def __hash__(self):
        return hash(self.hear)
    
    def available_actions(self):
        yield Action(listen=True)
        for i in range(len(self.hear)):
            yield Action(i)

class Action(POMDPAction):
    def __init__(self, direction=None, listen=False):
        if direction:
            self.d = direction
        self.listen = listen
    
    def __str__(self):
        if self.d:
            direction = "left" if self.d == LEFT else "right"
            return "open {} door".format(direction)
        elif self.listen:
            return "listening"
        else:
            return "nothing"
    
    def __eq__(self,o):
        return self.d == o.d and self.listen == o.listen

    def do_on(self, state):
        assert isinstance(state, State)
        r = 0
        if self.listen:
            r = -1
        else:
            assert self.d
            r = 10 if self.d == state.d else -100
        return (Observation(state.d), r)

class Tiger(DecisionProcess):
    def initial_belief(self):
        return random.choice([State(i) for i in range(2)])
        