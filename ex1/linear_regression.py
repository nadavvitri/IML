import pandas as pd
import numpy
import random


def pre_processing():
    data = pd.read_csv("kc_house_data2.csv")

    # can't be < 0 , e.g like price
    cols_to_clean = ["price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view",
                     "condition", "grade", "sqft_above", "sqft_basement", "yr_built", "yr_renovated", "zipcode",
                     "sqft_living15", "sqft_lot15"]
    cols_to_clean_zeros = ['sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'yr_built', 'sqft_living15', 'sqft_lot15']
    for col in cols_to_clean:
        data[col] = [None if i < 0 else i for i in data[col]]
    data[cols_to_clean_zeros] = data[cols_to_clean_zeros].replace(0, numpy.NaN)
    data.dropna(inplace=True)

    y = data.price
    # remove id, date and price columns
    data.drop(['id', 'date', 'price'], axis=1, inplace=True)
    data = pd.get_dummies(data, columns=["zipcode"], drop_first=True)  # One Hot encoding for zip code
    return data, y


def training_data():
    data, y = pre_processing()
    w = numpy.linalg.pinv(data).dot(y)
    size = data.shape[0] / 100
    for x in range(1, 100):
        data = data.sample(frac=1)
        train_data = data[:x * size]
        test_data = data[x * size:]



if __name__ == '__main__':
    training_data()
