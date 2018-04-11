import numpy as np
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

def create_matrices(filename, dummy=0):
    s = parse_data(filename)

    # Create Y matrix
    Y = []
    [Y.append(x.pop()) for x in s]

    # Create X and Y matrices
    X = np.matrix(s)
    # Make Y a column vector
    Y = np.matrix(Y).T

    # Add dummy column vector
    if dummy > 0:
        X = np.concatenate((np.matrix([1 for i in range(len(s))]).T, X), axis=1)

    return X, Y

# Calculate Sum of Squared error (SSE) with matrices
def SSE(w, X, y):
    return (y - X*w).T*(y - X*w)

# Alternative method for SSE
def _SSE(w, X, y):
    s = 0
    for j in range(y.shape[0]):
        s += (y[j]-np.matmul(w.T,X[j].T))**2

    return s

# Create matrices from training data
X, y = create_matrices(train_data_file, 1)

# Create weight column vector
w = inv(X.T * X) * X.T * y

# Print average sum error
print(SSE(w, X, y)/y.shape[0])

# Create matrices from test data
X, y = create_matrices(test_data_file, 1)

# Print average sum error
print(SSE(w, X, y)/y.shap[0])
