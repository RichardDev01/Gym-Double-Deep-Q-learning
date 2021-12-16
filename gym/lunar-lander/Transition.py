"""Schrijf een transitie class. Dit is een object met de SARSd (state, action, reward, next_state, done) tuple erin.
(Dataclass en NamedTuple lenen zich beide voor dit concept)"""
from dataclasses import dataclass


@dataclass
class Transition:
    state: object
    action: object
    reward: int
    next_state: object
    done: bool  # Dubbelop?
