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

if __name__ == "main":
	train_data = parse_data(training_data)
