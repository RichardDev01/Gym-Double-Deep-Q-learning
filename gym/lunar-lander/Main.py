"""Main file for Lunar landing using reinforcement learning."""

from Agent import Agent
from EpsilonGreedyPolicy import EpsilonGreedyPolicy
from Memory import Memory
from Transition import Transition
import gym


def train(episodes: int, batch_size: int, update_network_N: int = 10):
    """
    Train pytorch model using deep double Q-learning.

    Args:
        episodes:
        batch_size:
        update_network_N:

    Returns:

    """
    memory = Memory(size=1000)
    env = gym.make("LunarLander-v2")

    policy = EpsilonGreedyPolicy(env)

    state = env.reset()

    total_actions = env.action_space.n
    observation_length = len(state)

    agent = Agent(policy, alpha=0.1, tau=0.1, epsilon=0.1, batchsize=10, learning_rate=1,
                  model_input_size=observation_length, model_output_size=total_actions)

    agent.load_model('default_primary_name', 'default_target_name')

    # action = agent.get_action(state)
    # print(action)
    episode = 0
    update_network_counter = 1
    for i in range(episodes):
        episode += 1
        # Initialize S
        state = env.reset()
        done = False
        while not done:
            # Choose A from S using policy derived from Q (e.g., ε-greedy)
            # env.render()
            action = agent.get_action(state)

            # Take action A, observe R, S'
            next_state, reward, done, info = env.step(action)

            # add sarsa to memory
            memory.append_record(Transition(state, action, reward, done, next_state))

            state = next_state
            if memory.get_deque_len() >= batch_size:
                # batch = memory.sample(batch_size)

                if update_network_counter % update_network_N == 0:
                    agent.copy_model()
        # print(memory.sample())

        if episode % 10 == 0:
            print("saving")
            agent.approximator.save_network(agent.primary_network, agent.target_network)

    # print(agent.primary_network.parameters())
    # for param in agent.primary_network.parameters():
    #     print(param)


def evaluate(episodes):
    """
    Evaluate the target model for x episodes.

    Args:
        episodes:

    Returns:

    """
    env = gym.make("LunarLander-v2")

    policy = EpsilonGreedyPolicy(env)

    state = env.reset()

    total_actions = env.action_space.n
    observation_length = len(state)

    agent = Agent(policy, alpha=0.1, tau=0.1, epsilon=0, batchsize=10, learning_rate=1,
                  model_input_size=observation_length, model_output_size=total_actions)

    agent.load_model('default_target_name')

    for i in range(episodes):
        # Initialize S
        state = env.reset()
        done = False
        while not done:
            # Choose A from S using policy derived from Q (e.g., ε-greedy)
            env.render()
            action = agent.get_action(state)

            # Take action A, observe R, S'
            next_state, reward, done, info = env.step(action)


if __name__ == "__main__":
    train(episodes=500, batch_size=10, update_network_N=10)

    evaluate(episodes=10)
