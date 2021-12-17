"""Take random action in lunarlander v2 from Gym."""
import gym
env = gym.make('LunarLander-v2')

for i in range(100):
    state = env.reset()
    totalReward = 0

    for _ in range(100):
        env.render()
        # take a random action
        randomAction = env.action_space.sample()

        observation, reward, done, info = env.step(randomAction)
        print(len(observation))
        totalReward += reward

    print('Episode', i, ', Total reward:', totalReward)

env.close()
