import numpy as np
import csv
from math import exp

train_data_file = "usps-4-9-train.csv"
test_data_file = "usps-4-9-test.csv"

def parse_data(filename):
	data = []

	with open(filename, 'r') as f:
		reader = csv.reader(f)
		for line in reader:
			data.append(line)

	data = [[float(x) for x in t] for t in data]
	return data

def create_matrices(filename):
	data = parse_data(filename)

	#pop trailing 0s and 1s representing 4s and 9s
	Y = []
	[Y.append(x.pop()) for x in data]

	# make matrix of features
	X = np.matrix(data)

	# make matrix of results and Tranpose into a column
	Y = np.matrix(Y).T

	return X, Y

if __name__ == "__main__":
	X, Y = create_matrices(train_data_file)

	w = [0]*X.shape[0]
	w = np.matrix(w)

	for i in range(X.shape[0]):
		y_hat = 1/(1 + exp(w.T*X[i]*-1))
