import argparse
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

test_data_file = "housing_test.txt"
train_data_file = "housing_train.txt"

# Parse a data set from file 
def parse_data(filename):
    s = []
    # Get all lines from the file
    with open(filename) as f:
        s = f.readlines()
    # Remove line endings
    s = [x.strip().split() for x in s]
    # Ensure every value is a float
    s = [[float(x) for x in t] for t in s]

    return s

# Create matrices from data.
# dummy sets dummy column value
# random_features add n random features
def create_matrices(filename, dummy=False, random_features=0):
    s = parse_data(filename)

    # Create Y matrix
    Y = []
    [Y.append(x.pop()) for x in s]

    # Create X and Y matrices
    X = np.matrix(s)
    # Make Y a column vector
    Y = np.matrix(Y).T

    # Add dummy column vector
    if dummy:
        X = np.concatenate((np.matrix([1 for i in range(len(s))]).T, X), axis=1)

    if random_features > 0:
        for i in range(random_features):
            v = np.matrix([np.random.uniform(0, 250) for i in range(len(s))])
            X = np.concatenate((X, v.T), axis=1)

    return X, Y

def weight(X, y):
    return inv(X.T * X) * X.T * y

# Calculate Sum of Squared error (SSE) with matrices
def SSE(w, X, y):
    return np.asscalar((y - X*w).T*(y - X*w))

# Normalize the SSE by the number of examples
def ASE(w, X, y):
    return SSE(w, X, y)/y.shape[0]

def test_range(n):
    r = np.array([x*2 for x in range(1,n)])
    #r = np.append(r, [125, 150, 200, 250])

    n = r.size

    d = np.array([])
    f = np.array([])

    print("Creating graphs...", end='', flush=True)

    for i in range(n):
        X, y = create_matrices(train_data_file, 1, r[i])
        w = weight(X, y)
        d = np.append(d, ASE(w, X, y))
        X, y = create_matrices(test_data_file, 1, r[i])
        f = np.append(f, ASE(w, X, y))

    fig, ax = plt.subplots()

    ax.plot(r, d, label='Training')
    ax.plot(r, f, label='Testing')
    ax.legend(loc='upper left')
    fig.savefig("plot.png")
    print("Done", flush=True)

def perform_test(dummy=False):
    # Create matrices from training data
    X, y = create_matrices(train_data_file, dummy)
    # Create weight column vector
    w = weight(X, y)
    print("w vector:\n{}".format(w))
    # Print average sum error
    print("ASE for training: {}".format(ASE(w, X, y)))
    # Create matrices from test data
    X, y = create_matrices(test_data_file, dummy)
    # Print average sum error
    print("ASE for testing: {}".format(ASE(w, X, y)))
    print("")

if __name__ == "__main__":
    print("Dummy Column\n------------")
    perform_test(dummy=True)

    print("No Dummy Column\n---------------")
    perform_test(dummy=False)

    test_range(200)
