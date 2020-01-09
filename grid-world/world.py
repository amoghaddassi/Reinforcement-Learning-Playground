class World:
	"""Class for the environment of grid world. Will be a NxN grid with 2 terminal states"""
	def __init__(self, N, win_state = (0, 0), lose_state = (1, 0)):
		self.N = N
		self.x = int(self.N / 2)
		self.y = int(self.N / 2)
		self.win_state = win_state #reward +1 at this point
		self.lose_state = lose_state #reward -1 at this point
		
	def step(self, action):
		"""Returns a 3 tuple following the standard OpenAI Gym API:
		ret[0]: new state observation.
		ret[1]: reward
		ret[2]: done, true if the env has completed execution"""
		#compute new state
		observation = self.update_state(action)
		self.x, self.y = observation[0], observation[1]
		#compute reward
		reward = self.get_reward()
		#compute done flag
		if observation == self.win_state or observation == self.lose_state:
			done = True
		else:
			done = False
		return observation, reward, done

	def get_reward(self, curr_state = None):
		"""+- 1 reward at the terminal states. -.02 at all others."""
		if not curr_state:
			curr_state = (self.x, self.y)
		if curr_state == self.win_state: return 1
		elif curr_state == self.lose_state: return -1
		else: return -.02

	def move_state(self, action, curr_state = None):
		#0 - move left
		#1 - move right
		#2 - move up
		#3 - move down
		action = int(action)
		assert action in [0, 1, 2, 3], "Action space is Distinct(4)"
		if not curr_state:
			x, y = self.x, self.y
		else:
			x, y = curr_state[0], curr_state[1]
		if action == 0:
			x -= 1
		elif action == 1:
			x += 1
		elif action == 2:
			y += 1
		else:
			y -= 1
		#bounds checking to make sure we don't fall out of the world
		if x >= self.N or x < 0: x = self.x
		if y >= self.N or y < 0: y = self.y
		return (x, y)

	def test_move(self, move, curr_state):
		"""Returns the reward and next state from taking move in curr_state without changing anything.
		This is used as the agent's model of the environment assuming we have access to it."""
		next_state = self.move_state(move, curr_state)
		reward = self.get_reward(next_state)
		return next_state, reward

