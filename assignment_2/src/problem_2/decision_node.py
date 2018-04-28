class DecisionNode:
	# information gain
	gain = 0
	X = []
	y = []
	# index of X column that decided the split
	decision_index = None
	depth = 0
	# left and right child nodes
	child_left = None
	child_right = None

	def __init__(self, X, y, level):
		self.X = X
		self.y = y
		self.depth = level
