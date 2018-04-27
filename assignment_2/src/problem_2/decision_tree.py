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

if __name__ == "__main__":
	# load test data
	X, y = loadMatrices("../knn_test.csv")

	# normalize data in X
	X = normalize(X)
	print(X)
	print()
	print(y)
