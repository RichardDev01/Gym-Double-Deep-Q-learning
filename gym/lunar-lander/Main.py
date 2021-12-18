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
    env = gym.make("LunarLander-v2")

    policy = EpsilonGreedyPolicy(env)

    state = env.reset()

    total_actions = env.action_space.n
    observation_length = len(state)

    agent = Agent(policy, alpha=0.1, tau=0.1, epsilon=0.1, batchsize=10, learning_rate=1,
                  model_input_size=observation_length, model_output_size=total_actions)

    # Initialize primary network Q0, target network Q0', replay buffer D,t << 1
    agent.load_model('default_primary_name', 'default_target_name')
    memory = Memory(size=1000)

    episode = 0
    update_network_counter = 1
    # for each iteration do
    for i in range(episodes):
        done = False
        episode += 1

        # for each environment step do
        # Initialize S, observe first state
        state = env.reset()
        while not done:
            # Counter for checking if we update the networks
            update_network_counter += 1

            # env.render()

            # select at ~ π(at,st)
            action = agent.get_action(state)

            # Execute at and observe next state st+1 and reward rt = R(st,at)
            next_state, reward, done, info = env.step(action)

            # Store (st,at,rt,st+1) ~ D
            memory.append_record(Transition(state, action, reward, done, next_state))

            state = next_state

            # for each update step do
            if memory.get_deque_len() >= batch_size:
                if update_network_counter % update_network_N == 0:
                    # sample et = (st,at,rt,st+1) ~ D
                    batch = memory.sample(batch_size)

                    # Compute target Q value
                    # Perform gradient descent step on (Q*(st,at) - Q0(st,at))²
                    # Update target network parameters
                    # 0' ← t * 0 + (1 - t) * 0'
                    agent.train(batch)

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
