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

    def __init__(self, env: object, epsilon: float, decay_factor: float, minimal_epsilon: float):
        """Initialize epsilon greedy policy.

        :param env: The environment
        :param epsilon: Base epsilon
        :param decay_factor: Epsilon decay factor
        :param minimal_epsilon: Minimal epsilon
        """

        self.env = env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epsilon = epsilon
        self.decay_factor = decay_factor
        self.minimal_epsilon = minimal_epsilon

    def select_action(self, state: object, model: object):
        """Select the next action based on the state and policy.

        :param state: Observation of environment
        :param model: Neural network model
        :return: Chosen action
        """

        if np.random.rand(1)[0] > self.epsilon:
            with torch.no_grad():
                action = torch.argmax(model(torch.from_numpy(state).to(self.device))).item()
            return action
        else:
            return self.env.action_space.sample()  # Chooses a random action from all possible actions

    def epsilon_decay(self):
        """Add decay to epsilon overtime function optional."""

        if self.epsilon > self.minimal_epsilon:
            self.epsilon = self.epsilon * self.decay_factor
