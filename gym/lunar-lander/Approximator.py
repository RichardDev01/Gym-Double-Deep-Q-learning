"""file containing the class for saving, loading, creating and training a Double Q Neural Network.
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
    weights laden (pas belangrijk bij stap 10)
"""

import torch
from torch import nn, optim
import os


class Approximator:
    """Class for saving, loading, creating and training a Double Q Neural Network."""

    def __init__(self):
        """Create class variables for double deep learning."""
        self.q_network_1 = None
        self.q_network_2 = None
        self.model_path = os.path.dirname(os.getcwd()) + '/lunar-lander/models/'

        # setting device on GPU if available, else CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', self.device)
        self.optimizer = None
        self.loss_fn = nn.MSELoss()

    def set_optimizer(self, model: object, learning_rate: float):
        """Set the optimizer for the model.

        :param model: Neural network model
        :param learning_rate: Learning rate factor
        """
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def save_network(self,
                     primary_nn: object = None,
                     target_nn: object = None,
                     primary_nn_name: str = 'default_primary_name',
                     target_nn_name: str = 'default_target_name'):
        """Save the networks used for double Q-learning.

        :param primary_nn: Primary network object
        :param target_nn: Target network object
        :param primary_nn_name: Primary network file name
        :param target_nn_name: Target network file name
        """
        model_path = self.model_path

        # Save primary network if available
        if primary_nn is not None:
            PATH = model_path + primary_nn_name + ".pth"
            torch.save(self.q_network_1.state_dict(), PATH)

        # Save target network is available
        if target_nn is not None:
            PATH = model_path + target_nn_name + ".pth"
            torch.save(self.q_network_2.state_dict(), PATH)

    def load_network(self, primary_nn_name: str = 'default_primary_name.pth',
                     target_nn_name: str = 'default_target_name'):
        """Load networks used for double Q-learning.

        :param primary_nn_name: Primary network file name
        :param target_nn_name: Target network file name
        """
        model_path = self.model_path

        # Load primary network if none is present
        if self.q_network_1 is not None:
            PATH = model_path + (primary_nn_name + ".pth")
            self.q_network_1.load_state_dict(torch.load(PATH))
            self.q_network_1.eval()
            self.q_network_1.to(self.device)
            print(f"Succesfully loaded the primary network from: {PATH}")

        # Save target network is available
        if self.q_network_2 is not None:
            PATH = model_path + (target_nn_name + ".pth")
            self.q_network_2.load_state_dict(torch.load(PATH))
            self.q_network_2.eval()
            self.q_network_2.to(self.device)
            print(f"Succesfully loaded the target network from: {PATH}")

    def train_network(self, train_batch: object, primary_network: object, target_network: object, gamma: float):
        """Trains network.

        :param train_batch: Set of transitions
        :param primary_network: Primary network object
        :param target_network: Target network object
        :param gamma:Discount value for algorithm
        """

        state_batch = [x[0] for x in train_batch]
        action_batch = [x[1] for x in train_batch]
        reward_batch = [x[2] for x in train_batch]
        done_batch = [x[3] for x in train_batch]
        next_obs_batch = [x[4] for x in train_batch]

        state_batch = torch.stack(list(map(torch.tensor, state_batch))).to(self.device)
        next_obs_batch = torch.stack(list(map(torch.tensor, next_obs_batch))).to(self.device)

        with torch.no_grad():
            next_q_values = primary_network(next_obs_batch)
            next_q_values_target = target_network(next_obs_batch)

        # Q*(st,at) = rt +y * Q0'(st+1, argmax a' q0(st+1,a')
        q_star = torch.tensor([reward if done else
                               reward + gamma * q_target[torch.argmax(next_q_pred).item()]
                               for reward, next_q_pred, done, q_target in zip(reward_batch, next_q_values, done_batch, next_q_values_target)
                               ]).to(self.device)

        current_q_values = primary_network(state_batch)
        chosen_q = torch.stack([x[y] for x, y in zip(current_q_values, action_batch)]).to(self.device)
        loss = self.loss_fn(q_star.float(), chosen_q.float())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def create_network_q1(self, input_size: int = 8, output_size: int = 4, middle_layer_size: int = 12):
        """Create first neural network for the double Q learning.

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
        ).to(self.device)
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
        ).to(self.device)
        return self.q_network_2

    def get_network_info(self):
        """Print information about the networks used for double deep Q- Learning."""
        print(f"Primary network =\n{self.q_network_1}\nTarget network =\n{self.q_network_2}")
