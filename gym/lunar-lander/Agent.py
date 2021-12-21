"""This files contain the agent class used for Double Deep Q-Learning.

Schrijf nu een functie die train heet.

Dit is een functie van de agent die het policy-network update adhv het Double Deep Q-Learning algoritme.
Dit is een variatie op de Bellman Optimality Equation.
Het idee is dat je hierin targets aanmaakt voor je netwerk (je labels) dit gaat in de volgende stappen:

-Pak een batch transities uit het geheugen. De grootte van de batch is een hyperparameter
-Pak de next_state uit je transities en bereken adhv je policy netwerk wat de beste actie is
-Pak deze beste actie en bereken met je target netwerk wat de q-waarde van deze actie in next_state is.
Onthoud dat als next_state een terminal state is, alle q-waardes 0 moeten zijn.
-Bereken nu de target: reward + gamma * resultaat vorige stap
-Doe stap 2-5 voor alle transities in de batch
-Voor gradient descent uit alleen op de outputnodes van je netwerk die representatief zijn voor acties die daadwerkelijk genomen zijn.
-In pytorch kan je selecteren op welke output nodes je gradient descent wilt toepassen
-Als je library dit niet toestaat kan je zorgen dat je target een lijst is waarin alleen de index van de genomen actie
verandert naar de target berekend in stap 4
---------
"""
from Approximator import *


class Agent:
    """Agent class used for Double Deep Q-Learning."""

    def __init__(self, policy: object,
                 alpha: float = 0.1,
                 tau: float = 0.001,
                 batchsize: int = 10,
                 gamma: float = 1,
                 model_input_size: int = 8,
                 model_output_size: int = 4,
                 model_middle_layer_size: int = 12):
        """
        Initialize agent class.

        Er zijn nog allerlei eigenschappen van de agent niet gespecificeerd. Onder andere Gamma/Alpha/Batchsize//epsilon etc.
        Bedenk zelf waar die terecht moeten aan de hand van je theoretische begrip van het onderwerp.

        """
        self.alpha = alpha
        self.tau = tau
        self.batchsize = batchsize
        self.gamma = gamma
        self.policy = policy
        self.approximator = Approximator()
        self.primary_network = Approximator.create_network_q1(self.approximator, input_size=model_input_size,
                                                              output_size=model_output_size,
                                                              middle_layer_size=model_middle_layer_size)
        self.target_network = Approximator.create_network_q2(self.approximator, input_size=model_input_size,
                                                             output_size=model_output_size,
                                                             middle_layer_size=model_middle_layer_size)
        self.approximator.set_optimizer(self.primary_network)

    def load_model(self, primary_nn_name: str = 'default_primary_name', target_nn_name: str = 'default_target_name'):
        """
        Load pytorch model from models folder.

        Args:
            primary_nn_name:
            target_nn_name:

        Returns:
        """
        self.approximator.load_network(primary_nn_name, target_nn_name)
        self.primary_network = self.approximator.q_network_1
        self.target_network = self.approximator.q_network_2

    def get_action(self, state):
        """
        Get action from policy.

        Args:
            state:

        Returns:
        """
        # model = self.approximator.load_network()
        return self.policy.select_action(state, self.primary_network)

    def train(self, train_batch: object):
        """Train a network with the Double Deep Q-Learning algorithm."""
        # Train primary network
        # Compute target Q value
        # Perform gradient descent step on (Q*(st,at) - Q0(st,at))
        self.approximator.train_network(train_batch, self.primary_network, self.target_network, self.gamma)

        # Update target network
        # 0' ← t * 0 + (1 - t) * 0'
        self.copy_model()

    def copy_model(self):
        """
        Copy primary model weight to target model with tau value to update just a small bit.

        Schrijf een functie die copy_model heet, deze voegt de policy en het target-network samen.
            Tau is hiervoor een input parameter die bepaalt hoeveel procent van het target-netwerk vervangen wordt door het policy netwerk.
            We doen dit minder vaak dan het trainen van het policy netwerk.
            Maar dit gebeurt wel om de n stappen. N is een optimaliseerbare hyperparameter

        Returns:
        """
        # 0' ← t * 0 + (1 - t) * 0'
        for target, primary in zip(self.target_network.parameters(), self.primary_network.parameters()):
            target.data.copy_(self.tau * primary.data + (1.0 - self.tau) * target.data)

    def replay_memory(self):
        """Replay stuff."""
        pass
