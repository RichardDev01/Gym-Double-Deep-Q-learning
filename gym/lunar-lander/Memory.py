"""Schrijf een memory class. Deze heeft de volgende functies en eigenschappen:
    een size,
    een deque met transities
    een functie "sample" die een gedeelte van de memory ophaalt
    een record/append functie die nieuwe memories toevoegt"""
from collections import deque
import random


class Memory:
    def __init__(self, size: int):
        self.size = size
        self.transition_deque = deque(maxlen=size)

    def sample(self, batch_size: int = 10):
        """Returns a random memory sample"""
        return random.sample(self.transition_deque, batch_size)

    def append_record(self, record):
        """records a new memory record"""
        self.transition_deque.append(record)

    def get_deque_len(self):
        """Returns length of the deque"""
        return len(self.transition_deque)
