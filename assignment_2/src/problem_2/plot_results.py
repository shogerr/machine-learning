import csv
import numpy as np
import matplotlib.pyplot as plt

def parse_csv(filename):
    with open(filename, 'r',) as f:
        s = list(csv.reader(f))

    s = [[float(x) for x in t] for t in s]
    return s

def plot(col, name):
    fig, ax = plt.subplots()

    ax.plot(s[:,0], s[:,col], label='Accuracy')
    ax.legend(loc='lower right')
    ax.set_xlabel('iterations')
    ax.set_ylabel('accuracy')
    fig.savefig(name)

if __name__ == '__main__':
    s = np.matrix(parse_csv('accuracy.csv'))

    plot(1, 'train-accuracy.png')
    plot(2, 'test-accuracy.png')
