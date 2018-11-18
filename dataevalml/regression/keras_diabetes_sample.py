from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print("Diabetes Regression Sample with Keras")


# Load Diabetes data set
(data_X, data_y) = load_diabetes(True)

X_train, X_test, y_train, y_test = \
        train_test_split(data_X, data_y, test_size=.2, random_state=42)

# create model
model = Sequential()
model.add(Dense(12, input_dim=10, kernel_initializer='normal', activation='relu'))
model.add(Dense(20, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, kernel_initializer='normal', activation='relu'))
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
