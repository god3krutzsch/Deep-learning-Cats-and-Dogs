# zipfile - python module for extracting files from a zip file
import zipfile

# interact with os through python
import os
from os import makedirs
from os import listdir

# utility for copyfile
from shutil import copyfile

import numpy as np
import pandas as pd

# split training set
from sklearn.model_selection import train_test_split

# computer vision library for image processing.
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

# random number generator and ability to seed.
from random import seed
from random import random

# deep learning library, transform and augument image data
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# print(tf.__version__)
# print(tf.keras.__version__)

# 1.0 data acquisition and finding filenames and labels

train_zip_file_path = "/Users/godfreykrutzsch/Desktop/DL_Cat_Dog/train.zip"
test_zip_file_path = "/Users/godfreykrutzsch/Desktop/DL_Cat_Dog/test1.zip"

files = "/Users/godfreykrutzsch/Desktop/DL_Cat_Dog/"

# 1.0 image extraction

with zipfile.ZipFile(train_zip_file_path, 'r') as zipp:
    zipp.extractall(files)

with zipfile.ZipFile(test_zip_file_path, 'r') as zipp:
    zipp.extractall(files)

image_dir_train = "/Users/godfreykrutzsch/Desktop/DL_Cat_Dog/train/"
image_dir_test = "/Users/godfreykrutzsch/Desktop/DL_Cat_Dog/test1/"

# List all files in the directory
filenames = os.listdir(image_dir_train)
filenames_test = os.listdir(image_dir_test)
all_filenames = filenames + filenames_test

# find labels
labels = [x.split(".")[0] for x in filenames]
test_labels = [x.split(".")[0] for x in filenames_test]

# list all filenames and labels in DF
df_training_data = pd.DataFrame({"filename": filenames, "label": labels})
df_test_data = pd.DataFrame({"filename": filenames_test, "label": test_labels})

df_combined = pd.concat([df_training_data.reset_index(drop=True), df_test_data.reset_index(drop=True)], axis=0)
print("check shape of combine dataset")
print(df_combined.shape)

print("The shape of training data", df_training_data.shape)
print("The shape of test data", df_test_data.shape)

file_path_csv = "/Users/godfreykrutzsch/Desktop/DL_Cat_Dog/dump.csv"
df_combined.to_csv(file_path_csv, index=False)


# 2.0 standardise images for downstream machine learning with training data


def process_training_images():
    training_data = []

    for filename in filenames:
        file_path = os.path.join(image_dir_train, filename)

        # read image from that location with filename
        image = cv2.imread(file_path)

        if image is not None:
            image = cv2.resize(image, (150, 150))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            training_data.append([image])
        else:
            print("Test Resizing failed")


print("Set of training images have been standardised 150 x 150 and from BGR to RGB")


def process_testing_images():
    testing_data = []

    for filename in filenames_test:
        file_path = os.path.join(image_dir_test, filename)

        # read image from that location with filename
        image = cv2.imread(file_path)

        if image is not None:
            image = cv2.resize(image, (150, 150))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            testing_data.append([image, filename])
        else:
            print("Test Resizing failed")


print("Set of testing images have been standardised 150 x 150 and from BGR to RGB")

process_training_images()
process_testing_images()

# 3.0 split train test using data frame

# we use this to stratify and ensure the thus maintain class distribution (ratio) of cats and dogs
labels = df_training_data['label']

# split 80: 20
X_train, X_temp = train_test_split(df_training_data, test_size=0.2, stratify=labels, random_state=42)

# use to maintain class distribution with stratify the 20% ratio of cat and dogs as original data
label_test_val = X_temp['label']

# split 10: 10
X_test, X_val = train_test_split(X_temp, test_size=0.5, stratify=label_test_val, random_state=42)

print("The shape of train data", X_train.shape)
print("The shape of test data", X_test.shape)
print("The shape of validation data", X_val.shape)


# 4.0 exploratory data analysis

def visualise_subset_images():
    plt.figure(figsize=(9, 9))

    for i in range(20):
        plt.subplot(2, 10, i + 1)  # Create a subplot for each image
        file_path_local = os.path.join(image_dir_train, filenames[i])
        image = cv2.imread(file_path_local)
        if image is not None:  # Check if image is read correctly
            print("Image is ok")
            plt.imshow(image)
            plt.title(labels[i], fontsize=12)
            plt.axis('off')
        else:
            print(f"Error reading image: {filename}")

    plt.show()


visualise_subset_images()

# define the dataset directory on local machine

dataset_home = '/Users/godfreykrutzsch/Desktop/DL_Cat_Dog/dataset_dogs_vs_cats/'
subdirs = ['train/', 'test/']

# 5.0 setup directories structure on local machine

for subdir in subdirs:
    labeldirs = ['dogs/', 'cats/']
    for labeldir in labeldirs:
        newdir = os.path.join(dataset_home, subdir, labeldir)
        os.makedirs(newdir, exist_ok=True)

print("Check subdirectories")

# seed random number generator
seed(1)

# define ratio of pictures to use for validation
val_ratio = 0.2

# 6.0 copy training dataset images into subdirectories

src_directory = '/Users/godfreykrutzsch/Desktop/DL_Cat_Dog/train/'

# this loop goes through every file in source and constructs the full path to each file
for file in os.listdir(src_directory):
    src = src_directory + '/' + file

    # random assignment to training or testing
    dst_dir = 'train/'
    if random() < val_ratio:
        dst_dir = 'test/'

    # categorize and copy
    if file.startswith('cat'):
        dst = dataset_home + dst_dir + 'cats/' + file
        copyfile(src, dst)
    elif file.startswith('dog'):
        dst = dataset_home + dst_dir + 'dogs/' + file
        copyfile(src, dst)

# create an instance of ImageDataGenerator with rescaling
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

print(X_val.info)
X_val.head()

# global parameters for training
batch_size = 32
image_size = 150

# load training images from directory
train_generator = train_datagen.flow_from_dataframe(X_train,
                                                    directory='train/',  # directory containing training images
                                                    target_size=(image_size, image_size),
                                                    # Resize images to 150x 150 pixels
                                                    batch_size=batch_size,  # The number of images to return each batch
                                                    x_col='filename',
                                                    y_col='label'

                                                    )
