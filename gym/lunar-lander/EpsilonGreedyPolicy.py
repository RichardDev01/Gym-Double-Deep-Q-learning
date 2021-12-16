"""Schrijf een class EpsilonGreedyPolicy. De class heeft 1 noodzakelijke functie:
        select_action, deze heeft een state, mogelijke acties, een model a.k.a netwerk, en epsilon nodig
        Optioneel: decay functie die langzaam epsilon kleiner laat worden"""


class EpsilonGreedyPolicy:
    def __init__(self):
        pass

    def select_action(self, state, actions, model, epsilon):
        """Selects the next action based on the state and policy"""
        pass

    def epsilon_decay(self):
        """Optional"""
        pass
