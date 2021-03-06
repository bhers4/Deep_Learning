from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
import PIL

# Initialize Convolutional Neural Network
classifier = Sequential()

# Step 1 - Convolution
# Creates 32 feature maps with 3 rows and 3 columns
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

### Add second convolutional layer to increase accuracy
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
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

### Code from Keras Documentation
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('dataset/training_set',target_size = (64, 64),batch_size = 32,class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',target_size = (64, 64),batch_size = 32,class_mode = 'binary')
classifier.fit_generator(training_set,steps_per_epoch = 8000,epochs = 25,validation_data = test_set,validation_steps = 2000)
# End of Keras Code

print("End of CNN")

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'