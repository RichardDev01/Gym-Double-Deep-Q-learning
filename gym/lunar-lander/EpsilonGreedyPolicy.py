"""
File containing epsilon greedy policy.

Schrijf een class EpsilonGreedyPolicy. De class heeft 1 noodzakelijke functie:
select_action, deze heeft een state, mogelijke acties, een model a.k.a netwerk, en epsilon nodig
Optioneel: decay functie die langzaam epsilon kleiner laat worden
"""
import numpy as np
import torch


class EpsilonGreedyPolicy:
    """Epsilon greedy policy class."""

    def __init__(self, env):
        """Initialize epsilon greedy policy."""
        self.env = env

    def select_action(self, state: object, model: object, epsilon: float):
        """Select the next action based on the state and policy."""
        # print(state)
        if np.random.rand(1)[0] > epsilon:
            # action = torch.argmax(model(torch.from_numpy(state))).item()
            # print("Neural Network")
            return torch.argmax(model(torch.from_numpy(state))).item()
        else:
            # print("random")
            return self.env.action_space.sample()  # Chooses a random action from all possible actions

    def epsilon_decay(self):
        """Add decay to epsilon overtime function optional."""
        pass
