import gym
from randomAgent import RandomAgent
from simpleAgent import SimpleAgent

NUM_EPISODES = 5
env = gym.make('CartPole-v0')
env.reset()

agent = SimpleAgent()
action = 0

for ep in range(NUM_EPISODES):
	for i in range(1000):
		env.render()
		observation, reward, done, _ = env.step(action)
		action = agent.act(observation, reward, done)	
		if done:
			print("Failed on step %d" % i)
			break
env.close()
