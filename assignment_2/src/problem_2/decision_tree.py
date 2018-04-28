import numpy as np
from decision_node import *

MALIGNANT = 1
BENIGN = -1

# decision tree depth limits
d = [0, 1, 2, 3, 4, 5, 6]

# used for graph generation
node_count = 0

# loads data from text file with comma delimiter into numpy matrices
def loadMatrices(file_name):
	data = np.loadtxt(open(file_name, "rb"), delimiter=',').astype("float")

	y = data[:,0]
	X = data[:,1:]

	return X, y


def genGraphNode(node, parent_name):
	global node_count

	name = "a" + str(node_count)
	node_count += 1
	values = value_counter(node.y, ret_dict=True)

	output = name + " [label=\""
	
	if node.decision == MALIGNANT:
		output += "MALIGNANT"
	else:
		output += "BENIGN"

	output += " | " + str(values[MALIGNANT]) + " : " + str(values[BENIGN])
	output += "\"];\n"
	output += parent_name + " -> " + name + ";\n"

	if node.child_left != None:
		output += genGraphNode(node.child_left, name)
	if node.child_right != None:
		output += genGraphNode(node.child_right, name)
	return output


# generates a graph file to generate an image with graphviz
def genGraphFile(file_name, head_node):
	values = value_counter(head_node.y, ret_dict=True)
	main_label = "ROOT | " + str(values[MALIGNANT]) + " : " + str(values[BENIGN])

	# left and right label construction can be changed to recursive method
	# not worth the time now
	values = value_counter(head_node.child_left.y, ret_dict=True)
	if values[MALIGNANT] > values[BENIGN]:
		left_label = "MALIGNANT | "
	else:
		left_label = "BENIGN | "
	left_label += str(values[MALIGNANT]) + " : " + str(values[BENIGN])

	values = value_counter(head_node.child_right.y, ret_dict=True)
	if values[MALIGNANT] > values[BENIGN]:
		right_label = "MALIGNANT | "
	else:
		right_label = "BENIGN | "
	right_label += str(values[MALIGNANT]) + " : " + str(values[BENIGN])

	with open(file_name, 'w+')as f:
		f.write("digraph G {\n")
		f.write("main [label=\"" + main_label + "\"];\n")
		f.write(genGraphNode(head_node.child_left, "main"))
		f.write(genGraphNode(head_node.child_right, "main"))
		f.write("}")


# normalizes x matrix values to be between 0 and 1
def normalize(x):
	return x / x.max(axis=0)


# counts occurences of values
# returns list of each value / total
def value_counter(y, ret_dict=False):
	values = {}
	total = 0
	# count occurences of each value found in y
	for line in y:
		if line in values:
			values[line] += 1
		else:
			values[line] = 1
		total += 1

	# return values dictionary if param is set
	if ret_dict:
		return values

	# write values found / total values into a list
	results = []
	for key in values:
		if values[key] > 0:
			results.append(values[key]/float(total))

	return results


# performs uncertainty calculation
# y: numpy single column array
def entropy(y):
	# get value probabilities
	values = value_counter(y)

	# perform entropy calculation
	E = 0
	for p in values:
		E -= p * np.log2(p)

	return E


# runs benefit of split algorithm on y data set and its 2 branches
def info_gain(y, y1, y2):
	p1 = float(len(y1))/float(len(y))
	p2 = float(len(y2))/float(len(y))
	return entropy(y) - p1*entropy(y1) - p2*entropy(y2)


def binary_split(node, max_depth):
	# used to track the best information gain
	#node = DecisionNode()
	#node.y = y

	if node.depth >= max_depth:
		return node

	X = node.X
	y = node.y

	#iterate through columns in X
	for i in range(X.shape[1]):
		y_left, y_right = [], []
		X_left, X_right = [], []

		# iterate through rows in X
		for j in range(X.shape[0]):
			# split data based on X value
			if X[j][i] >= .5:
				y_right.append(y[j])
				X_right.append(X[j])
			else:
				y_left.append(y[j])
				X_left.append(X[j])

		# calculate gain with particular binary split
		gain = info_gain(y, y_left, y_right)
		# save binary split if it is better than other iterations
		if gain > node.gain:
			node.gain = gain
			node.decision_index = i
			node.child_left = DecisionNode(np.matrix(X_left).A, y_left, node.depth+1)
			node.child_right = DecisionNode(np.matrix(X_right).A, y_right, node.depth+1)

	# recurse down left and right side of tree
	binary_split(node.child_left, max_depth)
	binary_split(node.child_right, max_depth)

	return node


# runs decision tree algorithm
def train(file_name, max_depth):
	# load training data
	X, y = loadMatrices(file_name)

	# normalize data in X
	X = normalize(X)

	# create head/root node
	head_node = DecisionNode(X, y, 0)

	# create node based on best binary_split
	head_node = binary_split(head_node, max_depth)

	return head_node


# traveses generated tree recursivally
def tree_traversal(node, X):
	# if node is a leaf, return its decision
	if node.child_right == None or node.child_left == None:
		return node.decision

	if X[node.decision_index] >= .5:
		return tree_traversal(node.child_right, X)
	else:
		return tree_traversal(node.child_left, X)

# uses head_node tree generated from train to predict test data
def test(file_name, head_node):
	# load test data
	X, y = loadMatrices(file_name)

	X = normalize(X)

	total = X.shape[0]
	success = 0
	for i in range(X.shape[0]):
		result = tree_traversal(head_node, X[i])
		
		if result == y[i]:
			success += 1

	return success/total


if __name__ == "__main__":
	head_node = train("../knn_train.csv", d[2])

	#print gain from training
	print("Root node gain: " + str(head_node.gain))
	# generate graph file
	genGraphFile("stump.gv", head_node)

	#get success rate for test data
	succ_rate = test("../knn_test.csv", head_node)
	print("Test success rate: " + str(succ_rate))
	succ_rate = test("../knn_train.csv", head_node)
	print("Training success rate: " + str(succ_rate))
