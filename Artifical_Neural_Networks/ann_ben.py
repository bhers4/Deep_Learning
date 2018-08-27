
import numpy as np
from matplotlib import pyplot
import pandas as pd
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.layers import Dropout
from sklearn.model_selection import GridSearchCV

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

def build_classifier_optimizer_specific(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

# Read CSV
dataset = pd.read_csv('Churn_Modelling.csv')
# Take columns 3-12 put them into x
x = dataset.iloc[:,3:13].values
# Take columns 13 as our independent variable
y = dataset.iloc[:,13].values

# We have 2 columns that are strings so Label Encoder converts them into numbers
labelencoder_X_1 = LabelEncoder()
x[:, 1] = labelencoder_X_1.fit_transform(x[:, 1])
labelencoder_X_2 = LabelEncoder()
x[:, 2] = labelencoder_X_2.fit_transform(x[:, 2])

# One hot encoder
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]

# Split Data into Training and Validation Data, 80% Training and 20% Validation
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# Initializing Artifical Neural Network
classifier = Sequential()

# Dense Function Initialzes Weights to small numbers close to 0
# This line adds first hidden layer of 6 nodes
# For simplicity number of hidden layer nodes is average of input and output variables which is 11+1/2
classifier.add(Dense(units = 6, kernel_initializer='uniform' , activation='relu',input_dim=11))

# If you overfit you can use the dropout method This applies dropout to the first layer
#classifier.add(Dropout(p=0.1))

# Second hidden layer has 6 nodes, since it is the second hidden layer dont need input_dim = 11
classifier.add(Dense(units = 6, kernel_initializer='uniform' , activation='relu'))

# Output layer , only 1 output, need sigmoid function because output is 1 or 0
classifier.add(Dense(units = 1, kernel_initializer='uniform' , activation='sigmoid'))

# Create the ANN by compiling it
# Binary Cross entropy is binary logarthimc loss function, so we minimize this loss function
classifier.compile(optimizer='adam', loss='binary_crossentropy' , metrics=['accuracy'])

# Feeding our training dataset to our Artifical Neural Network
classifier.fit(x_train,y_train,batch_size=10 , epochs = 100)

y_pred = classifier.predict(x_test)
y_pred = (y_pred>0.5)
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#info = [0,0,600,1,40,3,60000,2,1,1,50000]
prediction = classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
print(prediction)
if (prediction>0.5):
    print("Leave")
else:
    print("Stay")


### K Fold Cross Validation of our Artifical Neural Networks
classifier = KerasClassifier(build_fn=build_classifier,batch_size=10 , epochs = 100)
# Accuracy is a list of K accuracies per each fold
accuracy = cross_val_score(estimator= classifier, X= x_train,y=y_train, cv = 10,verbose = 0,n_jobs=1)
print(accuracy)
length = len(accuracy)
i = 0
average = 0
for i in range(length):
    average = average+accuracy[i]
average = average/length
print(average)
means = np.mean(accuracy)
variance = np.std(accuracy)
print("Mean = " + str(means)+ " , and Variance = " + str(variance))



### Grid Search CV Class
classifier = KerasClassifier(build_fn=build_classifier)
# Parameters to tune
parameters = {'batch_size': [20,30,40],
              'nb_epoch':[150,200,250]}
# Create grid search object
grid_search = GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10)
# Fit Grid Search to our Artifical Neural Network
grid_search = grid_search.fit(x_train,y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print(best_parameters)
print(best_accuracy)


