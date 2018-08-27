from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

# Initialize Convolutional Neural Network
classifier = Sequential()

# Step 1 - Convolution
# Creates 32 feature maps with 3 rows and 3 columns
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Fully Connected Artificial Neural Networks
# Hidden Layer
classifier.add(Dense(units=128, activation = 'relu'))
# Output Layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))


# Create the ANN by compiling it
# Binary Cross entropy is binary logarthimc loss function, so we minimize this loss function
classifier.compile(optimizer='adam', loss='binary_crossentropy' , metrics=['accuracy'])

