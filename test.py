from sklearn.datasets import make_regression
import pandas as pd
from mylinereg import MyLineReg

X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]
test = MyLineReg(50, 0.1)
test.fit(X, y)
print(test.get_coef())
