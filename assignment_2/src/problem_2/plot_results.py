import csv
import numpy as np
import matplotlib.pyplot as plt

def parse_csv(filename):
    with open(filename, 'r',) as f:
        reader = csv.reader(f)
        x = []
        y = []
        for line in reader:
            x.append(int(line[0]))
            y.append(float(line[1]))

    return (x, y)

def plot(name):
    fig, ax = plt.subplots()

    ax.plot(test_results[0], test_results[1], label='Test Accuracy')
    ax.plot(train_results[0], train_results[1], label='Train Accuracy')
    ax.legend(loc='lower right')
    ax.set_xlabel('Max Depth')
    ax.set_ylabel('Accuracy')
    fig.savefig(name)

if __name__ == '__main__':
    test_results = parse_csv('test_rates.csv')
    train_results = parse_csv('train_rates.csv')

    plot('accuracy.png')
