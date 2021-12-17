"""Schrijf een class EpsilonGreedyPolicy. De class heeft 1 noodzakelijke functie:
        select_action, deze heeft een state, mogelijke acties, een model a.k.a netwerk, en epsilon nodig
        Optioneel: decay functie die langzaam epsilon kleiner laat worden"""
import numpy as np
import torch


class EpsilonGreedyPolicy:
    def __init__(self, env):
        self.env = env

    def select_action(self, state: object, model: object, epsilon: float):
        """Selects the next action based on the state and policy"""
        if np.random.rand(1)[0] < epsilon:
            return torch.argmax(model(state))
        else:
            return self.env.action_space.sample()  # Chooses a random action from all possible actions

    def epsilon_decay(self):
        """Optional"""
        pass



'''

class EpsilonSoftGreedyDoubleQPolicy(Policy):
    """Epsilon Soft greedy double Q policy."""

    def __init__(self, epsilon=0.9):
        """
        Create Epsilon Soft greedy double Q policy with parameters.
        :param epsilon: epsilon used in algorithm.
        """
        self.value_matrix = None
        self.epsilon = epsilon
        self.q_table_1 = None
        self.q_table_2 = None

    def decide_action(self, observation):
        """
        Decide action with highest value in q-table with a Epsilon change to take a random action.
        :param observation: observation is a dict containing information about the environment
        :return: Action chosen based on the observation
        """
        all_actions = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]

        if np.random.rand(1)[0] < self.epsilon:
            agent_pos = observation["agent_location"]
            chosen_action = np.argmax([x[0] + x[1] for x in zip(self.q_table_1[agent_pos[0]][agent_pos[1]], self.q_table_2[agent_pos[0]][agent_pos[1]])])
            return chosen_action
        else:
            return np.random.choice(all_actions)


'''