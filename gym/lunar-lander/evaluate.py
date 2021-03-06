"""Run the simulation with policy and model."""

from Agent import Agent
from EpsilonGreedyPolicy import EpsilonGreedyPolicy

import gym


def evaluate():
    """Evaluate the target model for x episodes."""

    env = gym.make("LunarLander-v2")

    policy = EpsilonGreedyPolicy(env, epsilon=1.0, decay_factor=0.995, minimal_epsilon=0.1)

    state = env.reset()

    total_actions = env.action_space.n
    observation_length = len(state)

    agent = Agent(policy, tau=0.1, batchsize=10, gamma=1,
                  model_input_size=observation_length, model_output_size=total_actions, model_middle_layer_size=256)

    agent.load_model('default_target_name', 'default_target_name')

    while True:
        # Initialize S
        state = env.reset()
        done = False
        while not done:
            # Choose A from S using policy derived from Q (e.g., ε-greedy)
            env.render()
            action = agent.get_action(state)

            # Take action A, observe R, S'
            next_state, reward, done, info = env.step(action)

            state = next_state


if __name__ == "__main__":

    evaluate()
