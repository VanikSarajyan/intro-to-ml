import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from my_linear_regression import MyLinearRegression

X = np.array([1,2,3,4,5,6]).reshape(-1,1)
y = np.array([3,5,7,9,11,13])

my_r = MyLinearRegression()
s_r = LinearRegression()
my_r.fit(X,y)
s_r.fit(X,y)

X_test = np.array([11,36,20]).reshape(-1,1)
my_pred = my_r.predict(X_test)
s_pred = s_r.predict(X_test)

print(my_pred)
print(s_pred)
