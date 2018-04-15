import pandas as pd
import numpy as np
import math
from matplotlib.pyplot import *


def pre_processing():
    data = pd.read_csv("kc_house_data.csv")

    # can't be < 0 , e.g like price
    cols_to_clean = ['price', 'bedrooms', 'bathrooms', 'sqft_lot', 'floors', 'waterfront', 'view',
                     'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
                     'sqft_living15', 'sqft_lot15']
    cols_to_clean_zeros = ['sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'yr_built', 'sqft_living15', 'sqft_lot15']
    for col in cols_to_clean:
        data[col] = [None if i < 0 else i for i in data[col]]
    data[cols_to_clean_zeros] = data[cols_to_clean_zeros].replace(0, np.NaN)
    data.dropna(inplace=True)

    # remove id and date columns and sqrt_living (linear independence sqft_above + sqft_basement = sqft_living)
    data.drop(['id', 'date', 'sqft_living'], axis=1, inplace=True)
    data = pd.get_dummies(data, columns=['zipcode'], drop_first=True)  # One Hot encoding for zip code
    return data


def graph(x, train_error, test_error):
    suptitle('Train and Test error (MSE)', fontweight="bold", fontsize=13)
    subplot(111)
    plot(x, train_error, x, test_error, '-')
    legend(('train error', 'test error'), loc='upper right')
    xlabel('x value', fontweight="bold")
    show()


def training_data():
    train_error, test_error, x_axis = [], [], []
    data = pre_processing()
    for x in range(1, 100):
        x_axis.append(x)
        # train data set and labels
        train = data.sample(frac=x/100)
        y_train = train.price
        train.drop(['price'], axis=1, inplace=True)

        # test data set and labels
        test = data.drop(train.index)
        y_test = test.price
        test.drop(['price'], axis=1, inplace=True)
        # model for prediction
        w = np.matmul(np.transpose(np.linalg.pinv(np.transpose(train))), y_train)

        # calculate train error by MSE
        train_predicted = train.dot(w)
        train_difference = train_predicted - y_train
        train_error.append(train_difference.mul(train_difference).mean())
        # calculate test error by MSE
        test_predict = test.dot(w)
        test_difference = test_predict - y_test
        test_error.append(test_difference.mul(test_difference).mean())

    graph(x_axis, train_error, test_error)


if __name__ == '__main__':
    training_data()
