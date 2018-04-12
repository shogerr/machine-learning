import numpy as np
import csv

train_data_file = "usps-4-9-train.csv"
test_data_file = "usps-4-9-test.csv"

def parse_data(filename):
	data = []

	with open(filename, 'r') as f:
		reader = csv.reader(f)
		for line in reader:
			data.append(line)
	return data

def create_matrices(filename):
	data = parse_data(filename)

	Y = []
	[Y.append(x.pop()) for x in data]

	X = np.matrix(data)
	Y = np.matrix(Y).T

	return X, Y

if __name__ == "__main__":
	X, Y = create_matrices(train_data_file)
