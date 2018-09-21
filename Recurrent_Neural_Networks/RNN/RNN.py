import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler


# Opens CSV into a DataFrame
dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
# We are training on opening stock price, the .values turns it into a np array
training_set = dataset_train.iloc[:,1:2].values

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values


# Feature Scaling
scaling = MinMaxScaler(feature_range=(0, 1))
### Scaling the Training Data
training_set_scaled = scaling.fit_transform(training_set)

# Creating Data Structure
X_train = []
y_train = []
for i in range(60, 1258):  # Time Step is 60
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

# Convert to np.arrays from lists
X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshaping for Keras RNN input... 3D Tensor with shape (batch size, timesteps,input_dims)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Start Building RNN

regressor = Sequential()
# Add LSTM Layer and Dropout Regularization to avoid overfitting
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1))) # units = 50, 50 "neurons"
regressor.add(Dropout(0.2))
# Add Second LSTM Layer
regressor.add(LSTM(units=50,return_sequences=True))  # Don't need input shape
regressor.add(Dropout(0.2))
# Add Third LSTM Layer
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))
# Add Fourth LSTM Layer
regressor.add(LSTM(units=50,return_sequences=False))  # return_sequences set to false since last lstm
regressor.add(Dropout(0.2))

# Output Layer
regressor.add(Dense(units=1))

# Compile the LSTM and specify loss function and optimizer
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fit LSTM to training data
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = scaling.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaling.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
