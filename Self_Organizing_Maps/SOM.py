import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone,pcolor,colorbar,plot,show
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
# Using Data from UCI Machine Learning Repository on Australian Credit Approval
# http://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)
'''
Using Self Organizing Maps to detect fraud in Credit Card Applications using Australian Credit Approval Dataset
'''
dataset = pd.read_csv("Credit_Card_Applications.csv")

x = dataset.iloc[:, :-1].values  # Everything except final row
y = dataset.iloc[:, -1].values  # Final Row

# Data isn't between 0 and 1 so we need MinMaxScaler again from sci-kit learn

scaler = MinMaxScaler(feature_range=(0, 1))
x_scaled = scaler.fit_transform(x) # Scales every column to be in between 0 and 1 through normalization

# For SOM someone already created an architecture for SOM's, using MiniSOM 1.0, license CC BY 3.0
# Training SOM
SOM = MiniSom(x=10, y=10, input_len=15)
SOM.random_weights_init(x_scaled)  # Initialize Weights
SOM.train_random(data=x_scaled, num_iteration=144)

bone()
pcolor(SOM.distance_map())
colorbar()
markers = ['o','s']
colors = ['r','g']
for i, x in enumerate(x_scaled):
    w = SOM.winner(x)
    plot(w[0]+0.5, w[1]+0.5, markers[y[i]], markeredgecolor=colors[y[i]], markerfacecolor='None', markersize=10, markeredgewidth=2)
show()

# Finding Anomalies
mappings = SOM.win_map(x_scaled)
# Change because number of potential frauds changes when you run it cause of stochastic nature
numberoftopcells = 4
helper = np.concatenate((SOM.distance_map().reshape(100,1),np.arange(100).reshape(100,1)),axis=1)
helper = helper[helper[:, 0].argsort()][::-1]
idx = helper[: numberoftopcells,1]
result_map = []
for i in range(12):
    for j in range(12):
        if (i*10+j) in idx:
            if len(result_map)==0:
                result_map = mappings[(i, j)]
            else:
                if len(mappings[(i,j)])>0:
                    result_map = np.concatenate((result_map,mappings[(i, j)]), axis=0)

frauds = scaler.inverse_transform(result_map)
print(len(frauds))
'''
    Adding Artifical Neural Network to go from unsupervised to supervised deep learning
'''

# Including Last column
customers = dataset.iloc[:, 1:]  # Customer list and information, don't include customer id

# Creating Dependent Variable - Last Column - Fraud or not answer
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1
    else:
        is_fraud[i] = 0

# Initialize ANN
sc = StandardScaler()
customers_scaled = sc.fit_transform(customers)
# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units=2, kernel_initializer = 'uniform', activation = 'relu', input_dim=15))
# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
classifier.fit(customers_scaled, is_fraud, batch_size = 1, epochs = 5)

# Predicting the probabilities of frauds
y_pred = classifier.predict(customers_scaled) # This makes a list of the probabilities of fraud
y_pred = np.concatenate((dataset.iloc[:, 0:1],y_pred),axis=1) # Makes 2 parallel columns
y_pred = y_pred[y_pred[:, 1].argsort()] # Sorts frauds and customer ID's together

print(y_pred)
