import numpy as np

# Learns a classifier for features x from a solution set S.
def knn_classifier(x, S, k, leave_out=0):
    distances = distance_list(x, S[:,1:])
    # Merge the prediction label with its distance.
    distances = np.array(list(zip(distances, S[:,0])))
    # Sort the list of 
    distances = distances[distances[:,0].argsort()]

    #return distances[0, 1]
    # Sum up votes.
    v = 0
    for i in range(k):
        v += distances[i, 1]

    # If the sum of the votes are positve or 0, then this is a
    # plus 1 classification. Vote counts of 0 are a tie and are a
    # positive classification.
    if v >= 0:
        return 1

    return -1

# Get a single distance.
def get_distance(x, x_):
    x = x.A1
    x_ = x_.A1
    return np.sqrt(np.dot(x-x_, x-x_))

# Get a list of distances.
def distance_list(x, S):
    distances = []
    for i in range(len(S)):
        distances.append(get_distance(x, S[i]))

    return distances

# Normalize each feature individually.
def normalize(X):
    return (X - X.min(0)) / X.ptp(0)

def normalize_features(S):
    return np.concatenate((np.matrix(S[:,0]).T, normalize(S[:,1:])), axis=1)

def calc_error(h, y):
    return (h != y).sum()/h.shape[0]
def count_error(h, y):
    return (h != y).sum()

# Get training set S and test set T.
S = np.loadtxt(open("data/knn_train.csv", "rb"), delimiter=',').astype("float")
T = np.loadtxt(open("data/knn_test.csv", "rb"), delimiter=',').astype("float")

# Take just the first column.
labels = S[:,0]

# Take all other columns.
S = normalize_features(S)

T = normalize_features(T)

error_training = []

# Predictions
for k in range(1, 70, 2):
    predictions = np.apply_along_axis(knn_classifier, 1, S[:,1:], S, k)
    predictions_test = np.apply_along_axis(knn_classifier, 1, T[:,1:], S, k)

    p = np.array(()).astype('int')
    for i in range(S.shape[0]):
        p = np.append(p, knn_classifier(S[i,1:], np.delete(S, i, 0), k))

    e_tr = calc_error(predictions, S[:,0].A1)
    e_te = calc_error(predictions_test, T[:,0].A1)
    e_cr = calc_error(p, S[:,0].A1)
    #e_tr = (predictions != S[:,0].A1).sum()/predictions.shape[0]
    #e_te = (predictions_test != S[:,0].A1).sum()/predictions_test.shape[0]

    error_training.append((k, e_tr, e_te, e_cr,
        count_error(predictions, S[:,0].A1),
        count_error(predictions_test, T[:,0].A1),
        count_error(p, S[:,0].A1)))

np.savetxt("knn_results.csv", np.asarray(error_training), delimiter=',')
