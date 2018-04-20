import numpy as np
import csv
import time
import sys

train_data_file = "usps-4-9-train.csv"
test_data_file = "usps-4-9-test.csv"

# exit condition
epsilon = 25
epoch_max = 200

# learning rate
eta = .001

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
    X = np.matrix([[x/255 for x in t] for t in data])
    X = np.concatenate((np.matrix([1 for i in range(len(data))]).T, X), axis=1)
    # make matrix of results and Tranpose into a column
    Y = np.matrix(Y).T

    return X, Y


# calculates sigmoid expression
# weight: one dimensional matrix row/column
# X: one dimensional matrix orientation opposite weight
def sigmoid(weight, X):
    return 1/(1 + np.exp(-1*np.dot(weight, X)))

# runs training algorithm from given data filename
def train(data_filename, regularize=False, find_accuracy=False, l=0):
    print(l)
    X, Y = create_matrices(data_filename)

    w = np.zeros(X.shape[1])

    iteration = 0
    accuracy_file = None
    if find_accuracy:
        accuracy_file = open("accuracy.csv", "w+")
    # pseudo do while loop
    while True:
        # reset gradient
        gradient = np.zeros(X.shape[1])
        for i in range(X.shape[0]):
            # A1 flattens X[i] to match shape of w.T
            y_hat = sigmoid(w.T, X[i].A1)

             #calculate change in gradient
            gradient = gradient + (y_hat - Y[i])*X[i]
            # perform regularization

        # modify weights with calculated gradient
        if regularize:
            w = w - eta * (gradient + l * w)
        else:
            w = w - eta * gradient

        # above calculation adds dimenision to w, needs to be flattened again
        w = w.A1

        if find_accuracy:
            train_iter_rate = test(train_data_file, w)
            iter_rate = test(test_data_file, w)
            accuracy_file.write(str(iteration) + ',' + str(train_iter_rate) + ',' + str(iter_rate) + '\n')

        iteration += 1
        # do while conditional
        #print(np.linalg.norm(gradient))
        if iteration > epoch_max:
            break

    if find_accuracy:
        accuracy_file.close()

    return w


# performs test using weights derived from train
# w: single dimenisional matrix should same orientation as X
def test(data_filename, w):
    X, Y = create_matrices(data_filename)

    total = 0
    success = 0
    for i in range(X.shape[0]):
        # run sigmoid function uses weights
        y_hat = sigmoid(w.T, X[i].A1)
        # check if it guessed correctly
        if y_hat >= .5 and Y[i] == 1:
            success += 1
        elif y_hat < .5 and Y[i] == 0:
            success += 1
        total += 1

    # calculate success rate
    rate = float(success)/float(total)
    return rate

if __name__ == "__main__":

    #X, Y = create_matrices(train_data_file)
    #print(X.shape)
    #sys.exit(0)

    # run training algorithm
    start_time = time.time()
    w = train(train_data_file, find_accuracy=True)
    print("Run time: " + str(time.time() - start_time))

    # perform tests on learned weights
    rate = test(test_data_file, w)
    print("success rate: " + str(rate*100))

    # perform tests for different values of lambda
    l_test = [10**-1, 1, 10**1, 10**2, 10**3]
    test_results = []
    train_results = []
    for l in l_test:
        print("lambda: " + str(l))
        w = train(train_data_file, regularize=True, l=l)
        test_results.append(test(test_data_file, w))
        train_results.append(test(train_data_file, w))

    with open('lamda_results.csv', 'w') as f:
        for i in range(len(l_test)):
            f.write(str(l_test[i]) + ',' + str(train_results[i]) + ',' + str(test_results[i]) + '\n')




