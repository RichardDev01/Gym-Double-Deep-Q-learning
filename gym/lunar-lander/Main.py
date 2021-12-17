from Agent import Agent
from Approximator import Approximator
from EpsilonGreedyPolicy import EpsilonGreedyPolicy
import gym


def train(episodes: int):

    env = gym.make("LunarLander-v2")

    policy = EpsilonGreedyPolicy(env)

    state = env.reset()

    total_actions = env.action_space.n
    observation_length = len(state)

    agent = Agent(policy, alpha=0.1, tau=0.1, epsilon=0, batchsize=10, learning_rate=1,
                  model_input_size=observation_length, model_output_size=total_actions)

    agent.load_model('default_primary_name', 'default_target_name')

    action = agent.get_action(state)
    print(action)
    # episode = 0
    for i in range(episodes):
        # Initialize S
        state = env.reset()
        done = False
        while not done:
            # Choose A from S using policy derived from Q (e.g., ε-greedy)
            env.render()
            action = agent.get_action(state)

            # Take action A, observe R, S'
            state, reward, done, info = env.step(action)


if __name__== "__main__":
    train(episodes = 100)
    # env = gym.make('LunarLander-v2')
    #
    # state = env.reset()
    #
    # # agent = Agent()
    #
    # approximator = Approximator()
    # #
    # approximator.create_network_q1()
    # #
    # approximator.create_network_q2()
    # #
    # approximator.get_network_info()
    #
    # approximator.save_network()
    #
    # approximator.load_network()
    #
    # approximator.get_network_info()