from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print("Wine Classification Sample with Keras\n")


# load data
(data_X, data_y) = load_wine(True)
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.33, random_state=42)

# create model
model = Sequential()
model.add(Dense(12, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
model.add(Dense(8, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model
model.fit(X_train, y_train, epochs=1000, batch_size=96, verbose=2)

# Evaluate Model
score = model.evaluate(X_train, y_train, verbose=0)
y_preds = model.predict(X_test)
mse = mean_squared_error(y_test, y_preds)
print('\nMSE: {0:0.4f}\n'.format(mse))

# Print Predictions
counter_true, counter_false = 0, 0
for y_pred, y_true in zip(y_preds, y_test):
    correct = y_true == round(y_pred[0])
    if correct:
        counter_true += 1
    else:
        counter_false += 1
    print("Pred: {0}, Actual: {1} -> Correct: {2}".format(y_pred, y_true, correct))

percent_correct = 100 * counter_true / (counter_true + counter_false)
print("\nPrediction evaluation:\n{0} Correct\n{1} Wrong\n -> {2:0.2f}% predicted correctly".format(counter_true,
                                                                                                   counter_false,
                                                                                                   percent_correct))
