import zipfile
import keras
from keras.api.models import Sequential, Model
from keras.api.layers import Dense, Dropout, Flatten
from keras.api.applications import VGG16
from keras.api.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from keras.api.preprocessing import image
import random
import tarfile
import urllib.request
import  os
import shutil


# Set seed for reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

## Part 1: Classification Problem: We need to classify the defect as dent or crack

# 1.1 Data Preparation
batch_size = 32
n_epochs = 5
img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 3)

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar"

tar_filename = 'aircraft_damage_dataset_v1.tar'
extracted_folder = "aircraft_damage_dataset_v1"

# Download the tar file
urllib.request.urlretrieve(url, tar_filename)
print(f"Downloaded {tar_filename}. Extraction will begin now.")

# Check if folder already exists
if os.path.exists(extracted_folder):
    print(f"The folder '{extracted_folder}' already exists. Removing the existing folder.")

    # Remove the existing folder to avoid overwriting or duplication
    shutil.rmtree(extracted_folder)
    print(f"Removed the existing folder: {extracted_folder}")

# Extract the contents of the tar file
with tarfile.open(tar_filename, 'r') as tar_ref:
    tar_ref.extractall() # this will extract to the current directory
    print(f"Extracted {tar_filename} successfully.")

# Define directories for train, test, and validation splits
extact_path = 'aircraft_damage_dataset_v1'
train_dir = os.path.join(extact_path, 'train')
test_dir = os.path.join(extact_path, 'test')
valid_dir = os.path.join(extact_path, 'valid')

## 1.2 Data preprocessing

# Create ImageDataGenerators to preprocess the data
# ImageDataGenerators: Provide an easy ah to augment(diversify) your images
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Next, we use flow_from_directory() method to load the images from directory and generate the training dataset.
# The flow_from_directory() method is part of the ImageDataGenerator class in Keras, and it plays a crucial role in automating the process of loading, preprocessing, and batching images for training, validation, and testing.
# We use the train_datagen object to load and preprocess the training images. Specifically, the flow_from_directory() function is used to read images directly from the directory and generate batches of data that will be fed into the model for training.

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    seed=seed_value,
    class_mode='binary',
    shuffle=True
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    class_mode='binary',
    seed=seed_value,
    batch_size=batch_size,
    shuffle=False,
    target_size=(img_rows, img_cols)
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    class_mode='binary',
    seed=seed_value,
    batch_size=batch_size,
    shuffle=False,
    target_size=(img_rows, img_cols)
)

## Model Definition
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(input_shape))
# VGG16: pre-trained model on the imagenet dataset
# weights: we specify to use the same weights as used for imagenet
# includeTop=False: removes the final fully connect classification layer, meaning we get the featrue extractor part
# input_shape: the shape of the input images
output = base_model.layers[-1].output
# gets the output of the last layer, the convolutional layer
output = keras.layers.Flatten()(output)

base_model = Model(base_model.input, output)
# creates a new Model  where the input is the original vgg16 input and the output is the flattened vector from the last conv

# Build the custom model
model = Sequential()
model.add(base_model)

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


## 1.4 Model training
model.fit(train_generator, epochs=n_epochs, validation_data=valid_generator)

# Access the training history
train_history = model.history.history  # After training

# plt.title("Training Loss")
# plt.ylabel("Loss")
# plt.xlabel('Epoch')
# plt.plot(train_history['loss'])
# plt.show()
#
# plt.title("Validation Loss")
# plt.ylabel("Loss")
# plt.xlabel('Epoch')
# plt.plot(train_history['val_loss'])
# plt.show()

## Visualizing predictions
def plot_image_with_title(image, model, true_label, predicted_label, class_names):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)

    # Convert labels from one-hot to class indices if needed, but for binary labels it's just 0 or 1
    true_label_name = class_names[true_label]
    pred_label_name = class_names[predicted_label]

    plt.title(f"True: {true_label_name}\nPredicted: {pred_label_name}")
    plt.axis('off')
    plt.show()

def test_model_on_image(test_generator, model, index_to_plot=0):
    test_images, test_labels = next(test_generator)

    # make predictions on the batch
    predictions = model.predict(test_images)

    # In binary classification, predictions are probabilities (float). Convert to binary (0 or 1)
    predicted_classes = (predictions > 0.5).astype(int).flatten()

    # Get the class indices from the test generator and invert them to get class names
    class_indices = test_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()} # Invert the dictionary

    # Specify the image to display based on the index
    image_to_plot = test_images[index_to_plot]
    true_labels = test_labels[index_to_plot]
    predicted_labels = predicted_classes[index_to_plot]

    plot_image_with_title(image=image_to_plot, model=model, true_label=true_labels, predicted_label=predicted_labels, class_names=class_names)


test_model_on_image(test_generator, model, 1)


