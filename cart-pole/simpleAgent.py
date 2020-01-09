class SimpleAgent:
	def __init__(self):
		pass
	
	def act(self, observation, reward, done):
		angle = observation[2]
		#If the pole points to the left, go left. Vice versa if pointing to the right.
		if angle > 0:
			return 1
		else:
			return 0
