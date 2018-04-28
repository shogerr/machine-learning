import numpy as np
import matplotlib.pyplot as plt

S = np.loadtxt(open("knn_results.csv", "rb"), delimiter=',').astype('float')

fig, ax = plt.subplots()
ax.plot(S[:,0], S[:,1], label="Training")
ax.plot(S[:,0], S[:,2], label="Test")
ax.plot(S[:,0], S[:,3], label="cross")
ax.legend(loc="lower right")
ax.set_xlabel('k')

fig.savefig("knn_results.png")
