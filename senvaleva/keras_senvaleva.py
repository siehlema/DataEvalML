from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# load data
dataset_train = numpy.loadtxt("../data/sensor_data_train")
X_train = dataset_train[:, 0:2]
y_train = dataset_train[:, 2:5]
dataset_test = numpy.loadtxt("../data/sensor_data_test")
X_test = dataset_test[:, 0:2]
y_test = dataset_test[:, 2:5]


# create model
model = Sequential()
model.add(Dense(20, input_dim=2, activation='relu', kernel_initializer='uniform'))
model.add(Dropout(0.2))
model.add(Dense(30, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(3, kernel_initializer='normal'))


# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


# Fit the model
model.fit(X_train, y_train, epochs=100, batch_size=96, verbose=2)


# Rate Model
scores = model.evaluate(X_train, y_train, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


# Calculate predictions
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('mse: {0}'.format(mse))

# Plot predictions
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.plot(y_pred[:, i], color='blue', linewidth=2)
    plt.plot(y_test[:, i], color='black', linewidth=2)
plt.show()
