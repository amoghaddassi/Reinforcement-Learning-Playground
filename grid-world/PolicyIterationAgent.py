import numpy as np
from world import World

class PolicyIterationAgent:
	def __init__(self, N, discount = .98):
		"""Initializes a random starting policy and a zero value vector."""
		self.N = N
		self.discount = discount
		self.world = World(self.N)
		#encoding the value fcn and policy as a 2d array
		self.value_array = np.zeros((self.N, self.N))
		self.policy = np.zeros((self.N, self.N))
		#populate the policy with random actions
		for i in range(self.N):
			for j in range(self.N):
				self.policy[i][j] = np.random.choice(range(4))

	def compute_policy(self, iter_tolerance = .01, max_iter = 1000):
		"""Runs the policy iteration algo till convergance starting from the current policy"""
		done = False
		runs = 0
		while not done and runs < max_iter:
			#policy evaluation to update the value function
			self.policy_eval(iter_tolerance)
			#policy improvement
			done = self.policy_improvement()		
			runs += 1

	def policy_eval(self, tolerance):
		"""Approximately solves the linear system given by the value array iteratively."""
		done = False
		while not done:
			delta = 0
			for i in range(self.N):
				for j in range(self.N):
					v = self.value_array[i][j]
					#compute which state current policy will place the agent
					move = self.policy[i][j]
					next_state, reward = self.world.test_move(move, (i, j))
					self.value_array[i][j] = reward + self.discount * self.value_array[next_state[0]][next_state[1]]
					delta = max(delta, abs(v - self.value_array[i][j]))
			if delta < tolerance: done = True			
	
	def policy_improvement(self):
		"""Using the current value array, computes the best greedy policy."""
		stable = True
		for i in range(self.N):
			for j in range(self.N):
				curr_action = self.policy[i][j]
				new_action = self.greedy_action((i, j))
				if new_action != int(curr_action): stable = False
		return stable

	def greedy_action(self, state):
		"""Given a state, returns the best move according to the current value array."""
		action = 0
		#check if moving left is valid, if so sets the value
		if state[0] > 0: value = self.value_array[state[0] - 1][state[1]]
		else: value = -1e38
		#considers a right move
		if state[0] < self.N - 1:
			if value < self.value_array[state[0] + 1][state[1]]:
				action = 1
				value = self.value_array[state[0] + 1][state[1]]
		#considers up move
		if state[1] < self.N - 1:
			if value < self.value_array[state[0]][state[1] + 1]:
				action = 2
				value = self.value_array[state[0]][state[1] + 1]
		#considers down move
		if state[1] > 0:
			if value < self.value_array[state[0]][state[1] - 1]:
				action = 3
				value = self.value_array[state[0]][state[1] - 1]
		return action

	def act(self, state):
		return self.policy[self[0]][self[1]]

