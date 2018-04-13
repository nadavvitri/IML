import pandas as pd
import numpy as np
from matplotlib.pyplot import *


def pre_processing():
    data = pd.read_csv("kc_house_data.csv")

    # can't be < 0 , e.g like price
    cols_to_clean = ["price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view",
                     "condition", "grade", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "zipcode",
                     "sqft_living15", "sqft_lot15"]
    cols_to_clean_zeros = ['sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'yr_built', 'sqft_living15', 'sqft_lot15']
    for col in cols_to_clean:
        data[col] = [None if i < 0 else i for i in data[col]]
    data[cols_to_clean_zeros] = data[cols_to_clean_zeros].replace(0, np.NaN)
    data.dropna(inplace=True)

    # remove id and date columns
    data.drop(['id', 'date'], axis=1, inplace=True)
    data = pd.get_dummies(data, columns=["zipcode"], drop_first=True)  # One Hot encoding for zip code
    return data


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
        # w = np.linalg.pinv(train).dot(y_train)
        # calculate train error by MSE
        y_head_train = train.dot(w)
        train_error.append(np.linalg.norm(y_head_train - y_train))
        # calculate test error by MSE
        y_head_test = test.dot(w)
        test_error.append(np.linalg.norm(y_head_test - y_test))

    suptitle('Graphs', fontweight="bold", fontsize=13)
    subplot(111)
    plot(x_axis, test_error, '-')
    xlabel('x value', fontweight="bold")
    ylabel('Train error (MSE)', fontweight="bold")
    show()


if __name__ == '__main__':
    training_data()
