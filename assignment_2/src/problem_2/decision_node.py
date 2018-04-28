class DecisionNode:
	gain = 0
	X = []
	y = []
	decision_index = None
	depth = 0
	# left and right will eventually be DecisionNodes themself
	child_left = ([], [])
	child_right = ([], [])

	def __init__(self, X, y, level):
		self.X = X
		self.y = y
		self.depth = level
