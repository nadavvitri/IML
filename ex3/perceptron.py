import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

############################################################
# Class definition
############################################################

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


############################################################
# Compare SVM and Perceptron from D1 and D2 distributions
############################################################

def generate_train_from_D1(m):
    x_train = np.random.multivariate_normal(np.zeros(2), np.identity(2), m)
    y_train = np.sign(x_train.dot(w))
    while len(np.unique(y_train)) <= 1:
        x_train = np.random.multivariate_normal(np.zeros(2), np.identity(2), m)
        y_train = np.sign(x_train.dot(w))
    return x_train, y_train


def generate_test_from_D1():
    x_test = np.random.multivariate_normal(np.zeros(2), np.identity(2), k)
    y_test = np.sign(x_test.dot(w))
    return x_test, y_test


def point_from_rec_one(m):
    x = 4 * np.random.random_sample((m, 1)) - 3
    y = 2 * np.random.random_sample((m, 1)) + 1
    return np.concatenate((x, y), axis=1)


def point_from_rec_minus_one(m):
    x = 4 * np.random.random_sample((m, 1)) - 1
    y = 2 * np.random.random_sample((m, 1)) - 3
    return np.concatenate((x, y), axis=1)


def generate_sets_from_rectangles(size):
    number_of_points_label_1 = np.random.binomial(size, 0.5)
    rec_one = point_from_rec_one(number_of_points_label_1)
    rec_one_label = np.full((number_of_points_label_1,), 1)

    rec_minus_one = point_from_rec_minus_one(size - number_of_points_label_1)
    rec_minus_one_label = np.full((size - number_of_points_label_1,), - 1)

    x_train = np.concatenate((rec_one, rec_minus_one), axis=0)
    y_train = np.concatenate((rec_one_label, rec_minus_one_label), axis=0)
    return x_train, y_train


def generate_train_from_D2(m):
    x_train, y_train = generate_sets_from_rectangles(m)
    if len(np.unique(y_train)) <= 1:
        x_train, y_train = generate_train_from_D2(m)
        return x_train, y_train
    return x_train, y_train


def generate_test_from_D2():
    x_test, y_test = generate_sets_from_rectangles(k)
    return x_test, y_test


def generate_train_from_D(m, d):
    if d == 1:
        return generate_train_from_D1(m)
    return generate_train_from_D2(m)


def generate_test_from_D(d):
    if d == 1:
        return generate_test_from_D1()
    return generate_test_from_D2()


def compare_svm_and_perceptron(d):
    svm_accuracy, perceptron_accuracy = [], []
    clf = SVC(C=1e10, kernel='linear')
    per = perceptron()

    for m in number_of_x_train:
        per_score, clf_score = 0, 0
        for i in range(repeat):
            # generate training and test sets with labels
            x_train, y_train = generate_train_from_D(m, d)
            x_test, y_test = generate_test_from_D(d)

            per.fit(x_train, y_train)
            clf.fit(x_train, y_train)

            per_score += per.score(x_test, y_test)
            clf_score += clf.score(x_test, y_test)

        perceptron_accuracy.append(per_score / repeat)
        svm_accuracy.append(clf_score / repeat)

    return perceptron_accuracy, svm_accuracy


def graphs(title, per_means, svm_means):
    fig = plt.figure()
    plt.title(title)
    plt.xlabel("number of train samples")
    plt.ylabel("mean accuracy")
    plt.plot(number_of_x_train, per_means, number_of_x_train, svm_means, marker='o')
    plt.legend(["perceptron", "SVM"], loc='upper right')
    return fig


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
    return x_train, y_train, x_test, y_test


def tpr_and_fpr(x_train, y_train, x_test, y_test):
    logistic = LogisticRegression()
    logistic.fit(x_train, y_train)

    probabilities = logistic.predict_proba(x_test)
    probabilities_sorted = np.argsort(probabilities[0])

    NP = sum(x > 0 for x in y_test)
    NN = y_test.shpae[0] - NP
    Ni, TPR = [], []

    for i in range(1, NP + 1):
        count, threshold = 0, 0
        for sample_id in probabilities_sorted[0]:
            if count == i:
                Ni.append(threshold)
                TPR.append((threshold - i) / NN)

            if y_test[sample_id] == 1:
                count += 1

            threshold += 1

def empirical_roc():
    for i in range(10):
        x_train, y_train, x_test, y_test = split_train_test()
        tpr_and_fpr(x_train, y_train, x_test, y_test)



    print 4

if __name__ == '__main__':
    empirical_roc()
    number_of_x_train = [5, 10, 15, 25, 70]
    k = 10000
    repeat = 500
    w = np.array([0.3, -0.5])

    per_means_D1, svm_means_D1 = compare_svm_and_perceptron(d=1)
    per_means_D2, svm_means_D2 = compare_svm_and_perceptron(d=2)

    title_figure_1 = "samples from Normal(0, I2)"
    title_figure_2 = "samples from D2"
    graphs(title_figure_1, per_means_D1, svm_means_D1)
    graphs(title_figure_2, per_means_D2, svm_means_D2)
    plt.show()
