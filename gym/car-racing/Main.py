"""Main file for Lunar landing using reinforcement learning."""

from Agent import Agent
from EpsilonGreedyPolicy import EpsilonGreedyPolicy
from Memory import Memory
from Transition import Transition
import gym
from torch.utils.tensorboard import SummaryWriter
import cv2
writer = SummaryWriter()


def process_state_image(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = state.astype(float)
    state /= 255.0
    return state


def train(episodes: int,
          batch_size: int,
          update_network_N: int = 10,
          tau: float = 0.01,
          gamma: float = 0.99,
          learning_rate: float = 0.0001,
          model_middle_layer_size: int = 256,
          memory_size: int = 10000,
          new_network: bool = False,
          base_epsilon: float = 0.9,
          decay_factor: float = 0.999,
          minimal_epsilon: float = 0.001
          ):
    """Train pytorch model using deep double Q-learning.

    :param episodes: number of episodes to train the policy
    :param batch_size: Memory batchsize for algorithm
    :param update_network_N: Network update episode step size
    :param tau: Variable for algorithm
    :param gamma: Discount value for algorithm
    :param learning_rate: learning rate factor for optimizer
    :param model_middle_layer_size: Number of hidden layer nodes
    :param memory_size: Maximum memory size
    :param new_network: To train a new network or train on the previous network
    :param base_epsilon: Base epsilon
    :param decay_factor: Epsilon decay factor
    :param minimal_epsilon: Minimal epsilon
    """

    env = gym.make('CarRacing-v0')

    policy = EpsilonGreedyPolicy(env, base_epsilon, decay_factor, minimal_epsilon)

    state = env.reset()

    total_actions = env.action_space
    observation_length = len(state)

    agent = Agent(policy, tau=tau, batch_size=batch_size, gamma=gamma, learning_rate=learning_rate)

    # Initialize primary network Q0, target network Q0', replay buffer D,t << 1
    if not new_network:
        agent.load_model('default_primary_name', 'default_target_name')

    memory = Memory(size=memory_size)

    update_network_counter = 1
    # for each iteration do
    for episode in range(episodes):

        done = False
        iteration = 0
        total_reward = 0

        # for each environment step do
        # Initialize S, observe first state
        state = env.reset()
        state = process_state_image(state)
        while not done:
            # Counter for checking if we update the networks
            update_network_counter += 1
            iteration += 1

            # select at ~ π(at,st)
            action = agent.get_action(state)

            # Execute at and observe next state st+1 and reward rt = R(st,at)
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            next_state = process_state_image(next_state)

            # Store (st,at,rt,st+1) ~ D
            memory.append_record(Transition(state, action, reward, done, next_state))

            state = next_state

            # for each update step do
            if memory.get_deque_len() >= batch_size:
                if update_network_counter % update_network_N == 0:
                    # sample et = (st,at,rt,st+1) ~ D
                    batch = memory.sample(batch_size)

                    # Update target network parameters
                    agent.train(batch)

        policy.epsilon_decay()

        writer.add_scalar('Total Reward', total_reward, episode)
        if episode % 10 == 0:
            print(f'Episode: {episode} - Total Reward: {total_reward} - Average Reward: {total_reward/iteration} - Epsilon: {policy.epsilon}')
            print("saving")
            agent.approximator.save_network(agent.primary_network, agent.target_network)
    writer.close()


if __name__ == "__main__":
    train(episodes=5000,
          batch_size=64,
          update_network_N=4,
          tau=0.001,
          gamma=0.99,
          learning_rate=0.1,
          model_middle_layer_size=256,
          memory_size=50000,
          new_network=False,
          base_epsilon=0.9,
          decay_factor=0.999,
          minimal_epsilon=0.001)

    # evaluate(episodes=1000)
