import numpy as np
from sklearn.svm import SVC


class perceptron:

    def __init__(self, X, y):
        self.number_of_samples = X.shape[0]
        self.weight_vector = np.zeros(X.shape[1])
        self.fit(X, y)

    def fit(self, X, y):
        """
        Calculate weight vector for prediction
        :param X: matrix of samples
        :param y: label vector for matrix x
        """
        flag = True
        while flag:
            for row in range(self.number_of_samples):
                if (y[row] * np.dot(self.weight_vector, X[row])) <= 0:
                    self.weight_vector += np.dot(y[row], X[row])
                else:
                    flag = False
                break

    def predict(self, x):
        """
        Gives prediction for sample x
        :param : sample x
        :return: label for x
        """
        return self.weight_vector * x

    def calculate_accuracy(self, x, y):
        accuracy = 0
        for row in x:
            if self.predict(row) != y:
                accuracy += 1
        return accuracy / 10000

def compare_svm_and_perceptron():
    number_of_training_points = [5, 10, 15, 25, 70]
    w = (0.3, -0.5)
    svm_accuracy, perceptron_accuracy = 0, 0
    clf = SVC(C=1e10, kernel='linear')
    for m in number_of_training_points:
        for i in range(1, 500):
            training_points = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], m)
            training_label_points = np.sign(training_points.dot(w))
            while -1 not in training_label_points or 1 not in training_label_points:
                training_points = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], m)
                training_label_points = np.sign(training_points.dot(w))

            test_points = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 10000)
            test_points_label = np.sign(test_points.dot(w))

            per = perceptron(training_points, training_label_points)
            perceptron_accuracy += per.calculate_accuracy(test_points, test_points_label) / i

            clf.fit(training_points, training_label_points)
            clf.predict(test_points)
            svm_accuracy += clf.score(test_points, test_points_label) / i


if __name__ == '__main__':
    compare_svm_and_perceptron()

