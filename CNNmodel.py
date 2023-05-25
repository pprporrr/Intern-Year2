import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

# Define the preprocessing parameters
target_size = (224, 224)
rescale_factor = 1.0 / 255

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=rescale_factor,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)

# Load and preprocess the training data
train_data = train_datagen.flow_from_directory(
    'path/to/training_data',
    target_size=target_size,
    batch_size=32,
    class_mode='categorical'
)

# Load and preprocess the testing data
test_datagen = ImageDataGenerator(rescale=rescale_factor)
test_data = test_datagen.flow_from_directory(
    'path/to/testing_data',
    target_size=target_size,
    batch_size=32,
    class_mode='categorical'
)

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Build the classification model
model = tf.keras.models.Sequential()
model.add(base_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(6, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, epochs=10, validation_data=test_data)

# Evaluate the model
loss, accuracy = model.evaluate(test_data)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

# Save the model for future use
model.save('path/to/saved_model')
