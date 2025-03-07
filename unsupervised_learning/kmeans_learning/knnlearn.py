import numpy as np

def eculidian_distance(x1,x2):
    return np.linalg.norm(x1-x2)

class KNN:
    def __init__(self,k):
        self.k = k

    def fit(self,X_train,Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self,x):
        prediction= [self._predict(i) for i in x]
        return prediction

    def _predict(self,x):
        distance = [eculidian_distance(x,x_train) for x_train in self.X_train]

        k_idicies = np.argsort(distance)[:self.k]

        k_nearest_labes = [self.Y_train[i] for i in k_idicies]

        most_common = np.bincount(k_nearest_labes).argmax()
        return most_common
