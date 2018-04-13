import numpy as np
import csv
import time

train_data_file = "usps-4-9-train.csv"
test_data_file = "usps-4-9-test.csv"

epsilon = .1
eta = 1

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

	empty_list = [0]*X.shape[0]
	w = np.matrix(empty_list)

	start_time = time.time()
	while True:
		gradient = np.matrix(empty_list)
		for i in range(X.shape[0]):
			y_hat = 1/(1 + np.exp(w.T*X[i]*-1))
			gradient = gradient + (y_hat - Y[i])*X[i].T # .T on X might be wrong but errors otherwise
		
		w = w - eta*gradient
		if gradient <= epsilon:
			break
	print(time.time() - start_time)
