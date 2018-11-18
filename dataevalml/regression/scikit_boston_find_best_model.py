"""Scikit-learn Module for Evaluation. Here a few different
regressors are used to find an optimal model for the boston housing dataset"""

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

print("Boston Housing Sample with multiple scikit learn models\n")

# Load Boston housing dataset
(data_X, data_y) = load_boston(True)

X_train, X_test, y_train, y_test = \
        train_test_split(data_X, data_y, test_size=.2, random_state=42)

regressors = [
    # ("SGD", SGDRegressor(max_iter=10000)),
    # ("SGDA", SGDRegressor(max_iter=1000, average=True)),
    ("Passive-Aggressive I", PassiveAggressiveRegressor(loss='epsilon_insensitive',
                                                         C=1.0, max_iter=100)),
    ("Passive-Aggressive II", PassiveAggressiveRegressor(loss='squared_epsilon_insensitive',
                                                          C=1.0, max_iter=100)),
    ("MLPRegressor", MLPRegressor(max_iter=500, hidden_layer_sizes=(100, 1000, 10), alpha=1, solver='adam')),
    ("LinearRegression", LinearRegression()),
    ("RandomForestRegressor", RandomForestRegressor(n_jobs=1, n_estimators=100))
]

(best_mse, best_regressor, best_name) = (99999, None, 'None')
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


print('\nBest Regressor is {0} with a MSE of {1}'.format(best_name, best_mse))
