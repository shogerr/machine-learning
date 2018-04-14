import numpy as np
import csv
import time
import sys

train_data_file = "usps-4-9-train.csv"
test_data_file = "usps-4-9-test.csv"

# exit condition
epsilon = .1
# learning rate
eta = .2

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
	#X = np.matrix(data)
	X = np.matrix([[x/255 for x in t] for t in data])

	# make matrix of results and Tranpose into a column
	Y = np.matrix(Y).T

	return X, Y

def sigmoid(weight, X):
	return 1/(1 + np.exp(-np.dot(weight, X)))

if __name__ == "__main__":
	X, Y = create_matrices(train_data_file)

	empty_list = [0]*X.shape[1]
	#w = np.matrix(empty_list)
	#w = np.matrix([np.random.uniform(0, 1) for i in range(X.shape[1])])
	w = np.zeros(256)

	start_time = time.time()
	# pseudo do while loop
	while True:
		# reset gradient
		gradient = np.matrix(empty_list)
		for i in range(X.shape[0]):
			#print("iteration: " + str(i))
			# A1 flattens X[i] to match w.T
			#y_hat = 1/(1 + np.exp(-np.dot(w.T, X[i].A1)))
			y_hat = sigmoid(w.T, X[i].A1)
			if y_hat >= .5:
				y_hat = 1

			gradient = gradient + (y_hat - Y[i])*X[i]
		
		w = w - (eta*gradient)
		w = w.A1 # might be incorrect but it makes it work. doesn't lose any data
		# do while conditional
		if np.linalg.norm(gradient) <= epsilon:
			break
	print("Run time: " + str(time.time() - start_time))
