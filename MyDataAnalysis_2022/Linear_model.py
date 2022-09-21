import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

boston = load_boston()
X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state= 1)

model=LinearRegression()
model.fit(X_train, y_train)
pred= model.predict(X_test)
print(pred)
plt.hist(y)
plt.show()