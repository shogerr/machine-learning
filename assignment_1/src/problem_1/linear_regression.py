import numpy as np
from numpy.linalg import inv

test_data_file = "housing_train.txt"
train_data_file = "housing_test.txt"

# data set
#
def parse_data(filename):
    s = []
    with open("housing_train.txt") as f:
        s = f.readlines()
    s = [x.strip().split() for x in s]
    s = [[float(x) for x in t] for t in s]
    return s

def create_matrices(dummy=0):
    s = parse_data(train_data_file)
    t = parse_data(test_data_file)

    # Create Y matrix
    Y = []
    [Y.append(x.pop()) for x in s]

    # Create our matrices
    X = np.matrix(s)
    Y = np.matrix(Y).T

    # Adding column vector of 1s
    if dummy > 0:
        X = np.concatenate((np.matrix([1 for i in range(len(s))]).T, X), axis=1)

    return X, Y

X, Y = create_matrices(1)

print(Y)
print(X)
#Z = np.matmul(X.T, Y)
#print(Z)
print(np.matmul(inv(np.matmul(X.T, X)), np.matmul(X.T, Y)))
#print(transpose(X))

