from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error

print("Boston Housing Regression Sample with TPot\n")
# Best pipeline: LassoLarsCV(ExtraTreesRegressor(PolynomialFeatures(Normalizer(input_matrix, norm=l2), degree=2,
# include_bias=False, interaction_only=False), bootstrap=False, max_features=0.35,
#  min_samples_leaf=1, min_samples_split=10, n_estimators=100), normalize=True)

# Load Boston housing dataset
(data_X, data_y) = load_boston(True)
X_train, X_test, y_train, y_test = \
    train_test_split(data_X, data_y, test_size=.2, random_state=42)

# TPot Regressor
tpot = TPOTRegressor(generations=10, population_size=100, verbosity=2)

# Start Training
tpot.fit(X_train, y_train)

# Print Scores
score = tpot.score(X_test, y_test)
y_pred = tpot.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Acc: {0:0.2f} - MSE: {1:0.4f}'.format(score, mse))
