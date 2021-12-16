"""Schrijf een memory class. Deze heeft de volgende functies en eigenschappen:
    een size,
    een deque met transities
    een functie "sample" die een gedeelte van de memory ophaalt
    een record/append functie die nieuwe memories toevoegt"""
from collections import deque


class Memory:
    def __init__(self, size: int, transition_deque: deque):
        self.size = size
        self.transition_deque = transition_deque

    def sample(self):
        """Returns a random memory sample"""
        pass

    def append_record(self):
        """records a new memory record"""
        pass
