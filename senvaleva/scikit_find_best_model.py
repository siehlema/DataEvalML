"""Scikit-learn Module for Sensor Value Evaluation. Here a few different
regressors are used to find an optimal model for the depending data"""

import numpy
from sklearn.cluster import FeatureAgglomeration
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline

dataset_train = numpy.loadtxt("../data/sensor_data_train")
X_train = dataset_train[:, 0:2]
y_train = dataset_train[:, 2:5]

dataset_test = numpy.loadtxt("../data/sensor_data_test")
X_test = dataset_test[:, 0:2]
y_test = dataset_test[:, 2:5]

regressors = [
    #    ("SGD", SGDRegressor(max_iter=10000)),
    #    ("SGDA", SGDRegressor(max_iter=1000, average=True)),
    #    ("MLPRegressor SGD", MLPRegressor(max_iter=5000, solver='sgd')),
    #("Passive-Aggressive I", PassiveAggressiveRegressor(loss='epsilon_insensitive',
    #                                                    C=1.0, max_iter=100)),
    #("Passive-Aggressive II", PassiveAggressiveRegressor(loss='squared_epsilon_insensitive',
    #                                                     C=1.0, max_iter=100)),
    ("MLPRegressor", MLPRegressor(max_iter=1000, hidden_layer_sizes=(20, 8, 12), alpha=1, solver='adam')),
    ("LinearRegression", LinearRegression()),
    ("RandomForestRegressor", RandomForestRegressor(n_jobs=1))
]

print(X_train[:, 0])
print(y_train[:])

(best_mse, best_regressor, best_name) = (1, None, 'None')
best_pred = None
for name, regr in regressors:
    regr.fit(X_train, y_train)

    score = regr.score(X_test, y_test)
    y_pred = regr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print('{0} - Acc: {1:0.2f} - MSE: {2:0.4f}'.format(name, score, mse))

    if mse < best_mse:
        (best_mse, best_regressor, best_name) = (mse, regr, name)
        best_pred = y_pred

    plt.plot(y_pred[:], color='blue', linewidth=2)
    plt.plot(y_test[:], color='black', linewidth=2)

    plt.show()

print('Best Regressor is {0} with a MSE of {1}'.format(best_name, best_mse))

plt.plot(best_pred[:, 0], color='blue', linewidth=2)
plt.plot(y_test[:, 0], color='black', linewidth=2)
plt.show()
