MALIGNANT = 1
BENIGN = -1

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
	decision = None

	def __init__(self, X, y, level):
		self.X = X
		self.y = y
		self.depth = level
		self.calcMajority()

	def calcMajority(self):
		counter = {}
		counter[MALIGNANT] = 0
		counter[BENIGN] = 0
		for line in self.y:
			counter[line] += 1

		if counter[MALIGNANT] > counter[BENIGN]:
			decision = MALIGNANT
		else:
			decision = BENIGN
