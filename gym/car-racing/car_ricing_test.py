"""Take random action in lunarlander v2 from Gym."""
import gym
env = gym.make('CarRacing-v0')

action_space = [
            (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), #           Action Space Structure
            (-1, 1,   0), (0, 1,   0), (1, 1,   0), #        (Steering Wheel, Gas, Break)
            (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Range        -1~1       0~1   0~1
            (-1, 0,   0), (0, 0,   0), (1, 0,   0)
        ]

for i in range(100):
    state = env.reset()
    totalReward = 0

    while True:
        env.render()
        # take a random action
        # randomAction = env.action_space.sample()

        forced_action = action_space[0]

        observation, reward, done, info = env.step(forced_action)

        print(observation)

        totalReward += reward
        if done:
            break
    print('Episode', i, ', Total reward:', totalReward)


env.close()