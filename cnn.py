# Convolutional Neural Network

# Part 1 - Building the CNN
# Importing Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the CNN
classifier = Sequential()

# Step 1 - Adding the first convolution layer
# Specify number of feature detectors and their dimensions, convert images to the same format, set activation function
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
# Apply max pooling to feature maps
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
# Flatten feature maps into a single vector
classifier.add(Flatten())

# Step 4 - Full connection
# Specity number of nodes and activation function for full connection (hidden) layer
classifier.add(Dense(units = 128, activation = 'relu'))

# Specify number of nodes and activation function for output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

# Augment images in training set
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2,  zoom_range = 0.2, horizontal_flip = True)

# Augment images in test set
test_datagen = ImageDataGenerator(rescale = 1./255)

# Create training set
training_set = train_datagen.flow_from_directory(r'C:\Users\firoj\Downloads\P16-Convolutional-Neural-Networks\Convolutional_Neural_Networks\chest_xray\train_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

# Create test set
test_set = test_datagen.flow_from_directory(r'C:\Users\firoj\Downloads\P16-Convolutional-Neural-Networks\Convolutional_Neural_Networks\chest_xray\test_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')

# Specify parameters to train the CNN
classifier.fit_generator(training_set, steps_per_epoch = 1000, epochs = 3, validation_data = test_set, validation_steps = 2000)


# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image 

# Load a test image to make a predicition for
test_image = image.load_img(r'C:\Users\firoj\Downloads\P16-Convolutional-Neural-Networks\Convolutional_Neural_Networks\chest_xray\single_prediction\pneumoniapositive.jpg', target_size = (64, 64))

# Convert image into a 3d-array
test_image = image.img_to_array(test_image)

# Add an additional dimension to 3-d array
test_image = np.expand_dims(test_image, axis = 0) 

# CNN makes prediction
result = classifier.predict(test_image)

# Determine which output value 0 and 1 correspond with
training_set.class_indices

if result[0][0] == 1:
    prediction = 'pneumonia positive'
else:
    prediction = 'pneumonia negative'
