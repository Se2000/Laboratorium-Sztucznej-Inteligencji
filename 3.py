import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# lin_reg = LinearRegression()
# lin_reg.fit(X, y)

# poly_reg = PolynomialFeatures(degree=6)
# X_poly = poly_reg.fit_transform(X)
# poly_reg.fit(X_poly, y)

# plt.scatter(X, y, color = 'red')
# plt.plot(X, lin_reg.predict(X), color = 'blue')
# plt.title('Truth or Bluff (Linear Regression)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()

# lin_reg_2 = LinearRegression()
# lin_reg_2.fit(X_poly, y)
#
# arr = np.array([6])
# arr = arr.reshape(1, -1)
# yu=lin_reg.predict(arr)
# zu=lin_reg_2.predict(poly_reg.fit_transform(arr))


# plt.scatter(X, y, color='red')
# plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()


# print(yu)
# print(zu)

df = pd.DataFrame({'Actual': X.flatten()})
df
