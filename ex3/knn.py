############################################################
# Class definition
############################################################

class perceptron:

    def __init__(self, k):
        self.number_nearest_neighbors = k
        
    def fit(self, X, y):
        """
        Calculate weight vector for prediction
        :param X: matrix of samples
        :param y: label vector for matrix x
        """
