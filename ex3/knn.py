import numpy as np
import pandas as pd
from collections import Counter

############################################################
# Class definition
############################################################


class knn:

    def __init__(self, k):
        self.number_nearest_neighbors = k
        self.x_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Simply stores the data
        :param X: matrix of samples
        :param y: label vector for matrix x
        """
        self.x_train = X
        self.y_train = y

    def predict(self, x):
        """
        Gives prediction for sample x
        :param : sample x
        :return: label for x
        """
        dist = np.linalg.norm(x - self.x_train, axis=1)
        distances = list(zip(dist, self.y_train))

        # sort by euclidean distance and save only the k nearest neighbors
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.number_nearest_neighbors]

        # max between k nearest neighbors labels
        counts = Counter(x[1] for x in neighbors)
        return max(counts, key=counts.get)


def split_train_test():
    data = pd.read_csv("spam.data", sep=" ", header=None)
    # test data set and labels
    x_test = data.sample(1000)
    y_test = x_test[len(x_test.columns) - 1]
    x_test.drop(x_test.columns[len(x_test.columns) - 1], axis=1, inplace=True)
    # train data set and labels
    x_train = data.drop(x_test.index)
    y_train = x_train[len(x_train.columns) - 1]
    x_train.drop(x_train.columns[len(x_train.columns) - 1], axis=1, inplace=True)
    return x_train.as_matrix(), y_train.as_matrix(), x_test.as_matrix(), y_test.as_matrix()


if __name__ == '__main__':
    k_values = [1, 2, 5, 10, 100]
    x_train, y_train, x_test, y_test = split_train_test()
    number_of_prediction = len(x_test)

    for k in k_values:
        correct = 0
        knn_classifier = knn(k)
        knn_classifier.fit(x_train, y_train)
        # calculate the classification accuracy
        for sample_id in range(number_of_prediction):
            if knn_classifier.predict(x_test[sample_id]) == y_test[sample_id]:
                correct += 1

        print("Classification accuracy for k = " + str(k) + ": " + str(correct / number_of_prediction))


