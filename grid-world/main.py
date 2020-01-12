#TODO: This agent/env loop is currently broken. Switching to gridworld2 folder to give this a fresh start.

from agent import Agent
from world import World
#Switches
N = 4
EPISODES = 1
ITERATIONS = 100

world = World(N)
agent = Agent(N)
agent.compute_policy()

reward_history = []
state_history = []
#agent env interaction loop
for ep in range(EPISODES):
	reward_history.append([])
	state_history.append([])
	for i in range(ITERATIONS):
		curr_state = world.state()
		action = agent.act(curr_state)
		observation, reward, done = world.step(action)
		reward_history[ep].append(reward)
		state_history[ep].append(observation)
		if done:
			break


		
