import numpy as np
from decision_node import *

MALIGNANT = 1
BENIGN = -1

# loads data from text file with comma delimiter into numpy matrices
def loadMatrices(file_name):
	data = np.loadtxt(open(file_name, "rb"), delimiter=',').astype("float")

	y = data[:,0]
	X = data[:,1:]

	return X, y

# generates a graph file to generate an image with graphviz
def genGraphFile(file_name, head_node):
	values = value_counter(head_node.y, ret_dict=True)
	main_label = "S | " + str(values[MALIGNANT]) + " : " + str(values[BENIGN])

	values = value_counter(head_node.child_left[1], ret_dict=True)
	if values[MALIGNANT] > values[BENIGN]:
		left_label = "MALIGNANT | "
	else:
		left_label = "BENIGN | "
	left_label += str(values[MALIGNANT]) + " : " + str(values[BENIGN])

	values = value_counter(head_node.child_right[1], ret_dict=True)
	if values[MALIGNANT] > values[BENIGN]:
		right_label = "MALIGNANT | "
	else:
		right_label = "BENIGN | "
	right_label += str(values[MALIGNANT]) + " : " + str(values[BENIGN])

	with open(file_name, 'w+')as f:
		f.write("digraph G {\n")
		f.write("main [label=\"" + main_label + "\"];\n")
		f.write("left [label=\"" + left_label + "\"];\n")
		f.write("main -> left;\n")
		f.write("right [label=\"" + right_label + "\"];\n")
		f.write("main -> right;\n")
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


def binary_split(X, y):
	# used to track the best information gain
	node = DecisionNode()
	node.y = y

	for i in range(X.shape[1]):
		y_left, y_right = [], []
		X_left, X_right = [], []

		for j in range(X.shape[0]):
			if X[j][i] >= .5:
				y_right.append(y[j])
				X_right.append(X[j])
			else:
				y_left.append(y[j])
				X_left.append(X[j])

		gain = info_gain(y, y_left, y_right)
		if gain > node.gain:
			node.gain = gain
			node.child_left = (X_left, y_left)
			node.child_right = (X_right, y_right)

	return node


# runs decision tree algorithm
def train(file_name):
	# load test data
	X, y = loadMatrices(file_name)

	# normalize data in X
	X = normalize(X)

	# create node based on best binary_split
	node = binary_split(X, y)
	print(node.gain)

	genGraphFile("stump.gv", node)


if __name__ == "__main__":
	train("../knn_train.csv")
