"""Module for Keras Evaluation. Right now this is a sample application
which uses the boston housing dataset for regression training"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
boston = load_boston()

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# load data
(data_X, data_y) = load_boston(True)
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.33, random_state=7)


# create model
model = Sequential()
model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
model.add(Dense(30, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, kernel_initializer='normal'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')


# Fit the model
model.fit(X_train, y_train, epochs=1000, batch_size=96, verbose=2)


# Rate Model
scores = model.evaluate(X_train, y_train, verbose=0)
print("\nAccuracy of Training: {0}".format(scores))


# Calculate predictions
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mse with Test Data: {0}'.format(mse))
