import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt


class perceptron:

    weight_vector = None

    def fit(self, X, y):
        """
        Calculate weight vector for prediction
        :param X: matrix of samples
        :param y: label vector for matrix x
        """
        flag = True
        rows, col = X.shape
        self.weight_vector = np.zeros(col)
        while flag:
            flag = False
            for row in range(rows):
                if y[row] * np.dot(self.weight_vector, X[row]) <= 0:
                    self.weight_vector += np.dot(y[row], X[row])
                    flag = True

    def predict(self, x):
        """
        Gives prediction for sample x
        :param : sample x
        :return: label for x
        """
        return np.sign(x.dot(self.weight_vector))

    def score(self, X, y, sample_weight=None):
        """
        Returns the mean accuracy on the given test data and labels.
        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.
        :param X: array-like, shape = (n_samples, n_features) Test samples.
        :param y: array-like, shape = (n_samples) or (n_samples, n_outputs) True labels for X.
        :param sample_weight : array-like, shape = [n_samples], optional Sample weights.
        :return score : float Mean accuracy of self.predict(X) wrt. y.
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


def generate_train_set(m):
        x_train = np.random.multivariate_normal(np.zeros(2), np.identity(2), m)
        y_train = np.sign(x_train.dot(w))
        while len(np.unique(y_train)) <= 1:
            x_train = np.random.multivariate_normal(np.zeros(2), np.identity(2), m)
            y_train = np.sign(x_train.dot(w))
        return x_train, y_train


def generate_test_set():
    x_test = np.random.multivariate_normal(np.zeros(2), np.identity(2), k)
    y_test = np.sign(x_test.dot(w))
    return x_test, y_test


def compare_svm_and_perceptron():
    svm_accuracy, perceptron_accuracy = [], []
    clf = SVC(C=1e10, kernel='linear')
    per = perceptron()

    for m in number_of_x_train:
        per_score, clf_score = 0, 0
        for i in range(repeat):
            # generate training and test sets with labels
            x_train, y_train = generate_train_set(m)
            x_test, y_test = generate_test_set()

            per.fit(x_train, y_train)
            clf.fit(x_train, y_train)

            per_score += per.score(x_test, y_test)
            clf_score += clf.score(x_test, y_test)

        perceptron_accuracy.append(per_score / repeat)
        svm_accuracy.append(clf_score / repeat)

    return perceptron_accuracy, svm_accuracy

def graphs():
    plt.title("samples from Normal(0, I2)")
    plt.xlabel("number of train samples")
    plt.ylabel("mean accuracy")
    plt.plot(number_of_x_train, per_means, number_of_x_train, svm_means)
    plt.legend(["perceptron", "SVM"], loc='upper right')
    plt.show()



if __name__ == '__main__':
    number_of_x_train = [5, 10, 15, 25, 70]
    k = 10000
    repeat = 500
    w = np.array([0.3, -0.5])
    per_means, svm_means = compare_svm_and_perceptron()
    graphs()
