from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
# Part two
from sklearn.tree import DecisionTreeRegressor
# Part three
from sklearn.ensemble import RandomForestRegressor

y=y.reshape(-1, 1)

sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

regressor=SVR(kernel='rbf')
regressor.fit(X, y)

plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X), max(X), 0.01
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.0]]))))

# Part two

regressor = DecisionTreeRegressor(random_state = 0)

regressor.fit(X, y)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

arr=np.array([6])
arr=arr.reshape(1,-1)
y_pred = regressor.predict(arr)

# Part three

regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

regressor.fit(X, y)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

arr=np.array([6])
arr=arr.reshape(1,-1)
y_pred = regressor.predict(arr)

