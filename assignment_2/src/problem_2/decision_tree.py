import numpy as np

# loads data from text file with comma delimiter into numpy matrices
def loadMatrices(file_name):
	data = np.loadtxt(open(file_name, "rb"), delimiter=',').astype("float")

	y = data[:,0]
	X = data[:,1:]

	return X, y


# normalizes x matrix values to be between 0 and 1
def normalize(x):
	return x / x.max(axis=0)


# counts occurences of values
# returns list of each value / total
def value_counter(y):
	values = {}
	total = 0
	# count occurences of each value found in y
	for line in y:
		if line in values:
			values[line] += 1
		else:
			values[line] = 1
		total += 1

	# write values found / total values into a list
	results = []
	for key in values:
		if values[key] > 0:
			results.append(values[key]/float(total))

	return results


# performs uncertainty calculation
# y: numpy single column array
def entropy(y):
	# get p1 and p2 probability
	values = value_counter(y)

	# perform entropy calculation
	E = 0
	for p in values:
		E -= p * np.log2(p)

	return E


def binary_split(X, y):
	for i in range(X.shape[1]):
		y_left, y_right = [], []
		for j in range(X.shape[0]):
			if X[j][i] >= .5:
				y_right.append(y[j])
			else:
				y_left.append(y[j])

		print(entropy(y_left), entropy(y_right))


def train(file_name):
	# load test data
	X, y = loadMatrices(file_name)

	# normalize data in X
	X = normalize(X)

	binary_split(X, y)


if __name__ == "__main__":
	train("../knn_train.csv")
