import numpy as np
import csv
import time

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

if __name__ == "__main__":
	X, Y = create_matrices(train_data_file)

	empty_list = [0]*X.shape[1]
	w = np.matrix(empty_list)
	#w = np.matrix([np.random.uniform(0, 1) for i in range(X.shape[1])])

	start_time = time.time()
	# pseudo do while loop
	while True:
		# reset gradient
		gradient = np.matrix(empty_list)
		for i in range(X.shape[0]):
			print(-w.T)
			print(X[i])
			print()
			print ((-w.T*X[i]))
			print (1 + (np.exp(-w.T*X[i]))) # always returns matrix full of 2s, exp is running as if arugment is all 0s
			y_hat = np.linalg.inv(1 + np.exp(-1*w.T*X[i])) # singular matrix error here
			print (y_hat.shape, Y[i].shape)
			gradient = gradient + (y_hat - Y[i])*X[i].T # .T on X might be wrong but errors otherwise
		
		w = w - (eta*gradient)
		# do while conditional
		if np.linalg.norm(gradient) <= epsilon:
			break
	print("Run time: " + time.time() - start_time)
