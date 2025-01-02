import pandas as pd
import numpy as np


class MyLineReg:
    def __init__(self, n_iter: int = 100, learning_rate: float = 0.1):
        self.iter = n_iter
        self.learning_rate = learning_rate
        self.weights = []

    def __str__(self):
        return f'MyLineReg class: n_iter={self.iter}, learning_rate={self.learning_rate}'

    def fit(self, x: pd.DataFrame, y: pd.Series, verbose=False):
        one_column = np.ones(len(x))
        x.insert(0, 'w0', one_column)
        self.weights = np.ones(x.shape[1])
        for i in range(self.iter):
            y_prediction = x.dot(self.weights)
            loss = np.mean((y - y_prediction)**2)
            grad = (2/len(y)) * np.dot(x.T, (y_prediction - y))
            self.weights = self.weights - grad * self.learning_rate
            if verbose:
                if i == 0:
                    print(f'start | loss:{loss}')
                elif i % verbose == 0:
                    print(f'{i} | loss: {loss}')

    def get_coef(self):
        return np.mean(self.weights[1::])
