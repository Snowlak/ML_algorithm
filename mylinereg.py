import pandas as pd
import numpy as np


class MyLineReg:
    def __init__(self, n_iter: int = 100, learning_rate: float = 0.1, metric=None):
        self.iter = n_iter
        self.learning_rate = learning_rate
        self.weights = []
        self.metric = metric
        self.loss = 0

    def __str__(self):
        return f'MyLineReg class: n_iter={self.iter}, learning_rate={self.learning_rate}'

    def fit(self, x: pd.DataFrame, y: pd.Series, verbose=False):
        one_column = np.ones(len(x))
        x.insert(0, 'x0', one_column)
        self.weights = np.ones(x.shape[1])
        y_prediction = pd.Series()
        for i in range(self.iter):
            y_prediction = x.dot(self.weights)
            self.loss = self.metric_error(y, y_prediction)
            grad = (2/len(y)) * np.dot(x.T, (y_prediction - y))
            self.weights = self.weights - grad * self.learning_rate
            if verbose:
                if i == 0:
                    print(f'start | loss:{self.loss}|{self.metric}: {self.loss}')
                elif i % verbose == 0:
                    print(f'{i} | loss: {self.loss}|{self.metric}: {self.loss}')
        self.loss = self.metric_error(y, x.dot(self.weights))

    def metric_error(self, y: pd.Series, y_prediction: pd.Series):
        if self.metric == 'mae':
            return np.mean(abs(y-y_prediction))
        elif self.metric == 'mse':
            return np.mean((y - y_prediction)**2)
        elif self.metric == 'rmse':
            return np.sqrt(np.mean((y - y_prediction)**2))
        elif self.metric == 'mape':
            return 100 * np.mean(abs((y-y_prediction)/y))
        elif self.metric == 'r2':
            return 1 - np.sum(((y-y_prediction)**2))/np.sum(((y-y.mean())**2))

    def get_best_score(self):
        return self.loss

    def get_coef(self):
        return np.mean(self.weights[1::])

    def predict(self, x: pd.DataFrame):
        one_column = np.ones(len(x))
        x.insert(0, 'x0', one_column)
        y_predict = x.dot(self.weights)
        return y_predict
