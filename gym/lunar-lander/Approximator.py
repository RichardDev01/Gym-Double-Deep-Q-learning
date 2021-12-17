"""file containing the class for saving, loading, creating and training a Double Q Neural Network."""

import torch
from torch import nn
from pathlib import Path
import os
"""
Schrijf een function approximator class. Dit is een neuraal netwerk.
Gebruik hiervoor een library naar keuze. De agent heeft twee instanties van approximators,
een policy-network en een target-network.
Begin met een Adam Optimizer met een learning rate van 0.001, RMS loss, en 2 hidden layers met 32 neuronen.
De class heeft de volgende functionaliteit:
    q-values teruggeven aan de hand van een state of lijst van states
    netwerk opslaan
    netwerk laden
    netwerk trainen
    weights handmatig zetten (pas belangrijk bij stap 10)
    weights laden (pas belangrijk bij stap 10)"""


class Approximator:
    """Class for saving, loading, creating and training a Double Q Neural Network."""

    def __init__(self):
        """Create class variables for double deep learning."""
        self.q_network_1 = None
        self.q_network_2 = None
        self.model_path = os.path.dirname(os.getcwd()) + '/lunar-lander/models/'


    def save_network(self, primary_nn_name: str = 'default_primary_name', target_nn_name: str = 'default_target_name'):
        """Save the networks used for double Q-learning."""

        # Models path
        model_path = self.model_path

        # Save primary network if available
        if self.q_network_1 is not None:
            PATH = model_path / (primary_nn_name + ".pth")
            torch.save(self.q_network_1.state_dict(), PATH)
            print(f"Succesfully saved the primary network as: {PATH}")

        # Save target network is available
        if self.q_network_2 is not None:
            PATH = model_path / (target_nn_name + ".pth")
            torch.save(self.q_network_2.state_dict(), PATH)
            print(f"Succesfully saved the target network as: {PATH}")

    def load_network(self, primary_nn_name: str = 'default_primary_name.pth', target_nn_name: str = 'default_target_name'):
        """Load networks used for double Q-learning."""  # TODO

        # Models path
        model_path = self.model_path

        # Load primary network if none is present
        if self.q_network_1 is not None:
            PATH = model_path + (primary_nn_name + ".pth")
            self.q_network_1.load_state_dict(torch.load(PATH))
            self.q_network_1.eval()
            print(f"Succesfully loaded the primary network from: {PATH}")

        # Save target network is available
        if self.q_network_2 is not None:
            PATH = model_path + (target_nn_name + ".pth")
            self.q_network_2.load_state_dict(torch.load(PATH))
            self.q_network_2.eval()
            print(f"Succesfully loaded the target network from: {PATH}")


    def train_network(self):
        """Train network."""  # TODO
        pass

    def set_weights(self):
        """Set weights."""  # TODO
        pass

    def load_weights(self):
        """Load weights."""  # TODO
        pass

    def create_network_q1(self, input_size: int = 8, output_size: int = 4, middle_layer_size: int = 12):
        """
        Create first neural network for the double Q learning.

        The network is build out of 4 layers in total from
        which 2 are hidden. This network is the primary network.

        :param input_size: Input size for the first layer
        :param output_size: Output size for the last layer
        :param middle_layer_size: Middle layer size of the first hidden layer
        """
        # Check if middle layer is smaller then input size and correct it
        if middle_layer_size < input_size:
            middle_layer_size = int(round(input_size * 1.5))
            print(
                f"Middle layer is smaller then input size and this is not ideal. check middle layer to size {middle_layer_size} ")

        # Calculate the output size of the 2e middle layer to be a bit smaller then the previous
        middle_layer_output = int(round(middle_layer_size // 1.25))

        self.q_network_1 = nn.Sequential(
            nn.Linear(input_size, middle_layer_size),
            nn.ReLU(),
            nn.Linear(middle_layer_size, middle_layer_output),
            nn.ReLU(),
            nn.Linear(middle_layer_output, output_size),
        )
        return self.q_network_1

    def create_network_q2(self, input_size: int = 8, output_size: int = 4, middle_layer_size: int = 12):
        """
        Create second neural network for the double Q learning.

        The network is build out of 4 layers in total from
        which 2 are hidden. This network is the target network

        :param input_size: Input size for the first layer
        :param output_size: Output size for the last layer
        :param middle_layer_size: Middle layer size of the first hidden layer
        """
        # Check if middle layer is smaller then input size and correct it
        if middle_layer_size < input_size:
            middle_layer_size = int(round(input_size * 1.5))
            print(
                f"Middle layer is smaller then input size and this is not ideal. check middle layer to size {middle_layer_size} ")

        # Calculate the output size of the 2e middle layer to be a bit smaller then the previous
        middle_layer_output = int(round(middle_layer_size // 1.25))

        self.q_network_2 = nn.Sequential(
            nn.Linear(input_size, middle_layer_size),
            nn.ReLU(),
            nn.Linear(middle_layer_size, middle_layer_output),
            nn.ReLU(),
            nn.Linear(middle_layer_output, output_size),
        )
        return self.q_network_2

    def get_network_info(self):
        """Print information about the networks used for double deep Q- Learning."""
        print(f"Primary network =\n{self.q_network_1}\nTarget network =\n{self.q_network_2}")
