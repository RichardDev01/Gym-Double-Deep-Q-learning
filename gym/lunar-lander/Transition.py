"""
Dataclass file used for keeping track of transitions for the memory.

Schrijf een transitie class. Dit is een object met de SARSd (state, action, reward, next_state, done) tuple erin.
(Dataclass en NamedTuple lenen zich beide voor dit concept)
"""
from dataclasses import dataclass


@dataclass
class Transition:
    """Class object for keeping track of transitions."""

    state: object
    action: object
    reward: int
    done: bool  # Dubbelop?
    next_state: object
