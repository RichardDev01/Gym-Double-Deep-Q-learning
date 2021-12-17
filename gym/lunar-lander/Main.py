from Agent import Agent

from Approximator import Approximator

# agent = Agent()

approximator = Approximator()

approximator.create_network_q1()

approximator.create_network_q2()

approximator.get_network_info()

approximator.save_network()
