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

# Phase 1

# 1.0 data acquisition and finding filenames and labels

train_zip_file_path = "/Users/godfreykrutzsch/Desktop/DL_Cat_Dog/train.zip"
test_zip_file_path = "/Users/godfreykrutzsch/Desktop/DL_Cat_Dog/test1.zip"

files = "/Users/godfreykrutzsch/Desktop/DL_Cat_Dog/"

# 1.1 image extraction

with zipfile.ZipFile(train_zip_file_path, 'r') as zipp:
    zipp.extractall(files)

with zipfile.ZipFile(test_zip_file_path, 'r') as zipp:
    zipp.extractall(files)

image_dir_train = "/Users/godfreykrutzsch/Desktop/DL_Cat_Dog/train/"
image_dir_test = "/Users/godfreykrutzsch/Desktop/DL_Cat_Dog/test1/"

# 1.2 List all files in the directory
filenames_training = os.listdir(image_dir_train)
filenames_test = os.listdir(image_dir_test)
all_filenames = filenames_training + filenames_test

# 1.3 Find historical labels
labels_training = [x.split(".")[0] for x in filenames_training]
test_labels = [x.split(".")[0] for x in filenames_test]

# 1.4 define new labels without the .dot in the file name. remember the test data does not have lables do not apply
new_filename_training = [x.replace('.'[0], '_',1) for x in filenames_training]


# 1. 5 list all filenames and labels in DF
df_training_data = pd.DataFrame({"filename": new_filename_training, "label": labels_training})
df_test_data = pd.DataFrame({"filename": filenames_test, "label": test_labels})

# 1.6 option to convert dog and cats to 1 and 2.

df_combined = pd.concat([df_training_data.reset_index(drop=True), df_test_data.reset_index(drop=True)], axis=0)
print("check shape of combine dataset")
print(df_combined.shape)

print("The shape of training data", df_training_data.shape)
print("The shape of test data", df_test_data.shape)
print("", df_training_data)
print("", df_test_data)

file_path_csv = "/Users/godfreykrutzsch/Desktop/DL_Cat_Dog/dump.csv"
df_combined.to_csv(file_path_csv, index=False)


# Phase 2

# 2.1 standardise images for downstream machine learning with training data


def process_training_images():
    training_data = []

    for filename in filenames_training:
        file_path = os.path.join(image_dir_train, filename)

        # read image from that location with filename
        image = cv2.imread(file_path)

        if image is not None:
            image = cv2.resize(image, (150, 150))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            training_data.append([image])
        else:
            print("Training Resizing failed")


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
            testing_data.append([image])
        else:
            print("Test Resizing failed")


print("Set of testing images have been standardised 150 x 150 and from BGR to RGB")

process_training_images()
process_testing_images()

# Phase 3

# 3.1 split train test using data frame

# we use this to stratify and ensure the thus maintain class distribution (ratio) of cats and dogs
labels = df_training_data['label']

# split 80: 20
X_train, X_temp = train_test_split(df_training_data, test_size=0.2, stratify=labels, random_state=42)

# use to maintain class distribution with stratify the 20% ratio of cat and dogs as original data
label_test_val = X_temp['label']

# split 10: 10
X_test, X_val = train_test_split(X_temp, test_size=0.5, stratify=label_test_val, random_state=42)

print("The shape of train data", X_train.shape)
print(X_train.info())
print("The shape of test data", X_test.shape)
print(X_test.info())
print("The shape of validation data", X_val.shape)
print(X_val.info())


# 4.0 exploratory data analysis

def visualise_subset_images():
    plt.figure(figsize=(9, 9))

    for i in range(20):
        plt.subplot(2, 10, i + 1)  # Create a subplot for each image
        file_path_local = os.path.join(image_dir_train, filenames_training[i])
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

# Phase 3

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

# 6.0 copy training dataset images into subdirectories we have created from the below directory
src_directory = '/Users/godfreykrutzsch/Desktop/DL_Cat_Dog/train'

# this loop goes through every file in source and constructs the full path to each file
for file in os.listdir(src_directory):
    src = src_directory + '/' + file
    #print(src)

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

# lets check.
path1 = "/Users/godfreykrutzsch/Desktop/DL_Cat_Dog/dataset_dogs_vs_cats/train/dogs"
path2 = "/Users/godfreykrutzsch/Desktop/DL_Cat_Dog/dataset_dogs_vs_cats/train/cats"
path3 = "/Users/godfreykrutzsch/Desktop/DL_Cat_Dog/dataset_dogs_vs_cats/test/dogs"
path4 = "/Users/godfreykrutzsch/Desktop/DL_Cat_Dog/dataset_dogs_vs_cats/test/cats"

print("The number of dogs in training", len(os.listdir(path1)))
print("The number of cats in training", len(os.listdir(path2)))
print("The number of dogs in test/validation", len(os.listdir(path3)))
print("The number of cats in test/ validation", len(os.listdir(path4)))

# create an instance of ImageDataGenerator with rescaling
train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# train_datagen = ImageDataGenerator()
# val_datagen = ImageDataGenerator()
# test_datagen = ImageDataGenerator()

print("check xtrain, xvale and xtrain values")

print("X_val")

print(X_val.info)
X_val.head()
print(X_val.shape)

print("X test")

print(X_test.info)
X_test.head()
print(X_test.shape)

print("X train")

print(X_train.info)
X_train.head()
print(X_train.shape)

# global parameters for training
batch_size = 32
image_size = 150



# load training images from directory
train_generator = train_datagen.flow_from_dataframe(X_train,
                                                    directory="/Users/godfreykrutzsch/Desktop/DL_Cat_Dog/dataset_dogs_vs_cats/train/",  # directory containing training images
                                                    x_col='filename',
                                                    y_col='label',
                                                    batch_size=batch_size,  # The number of images to return each batch
                                                    validate_filenames=False,
                                                    target_size=(image_size, image_size),
                                                    color_mode='rgb',
                                                    )
validation_generator = val_datagen.flow_from_dataframe(X_val,
                                                       directory='train/',
                                                       x_col='filename',
                                                       y_col='label',
                                                       batch_size=batch_size,
                                                       validate_filenames=False,
                                                       target_size=(image_size, image_size),
                                                       shuffle=False,
                                                       color_mode='rgb')

test_generator = test_datagen.flow_from_dataframe(X_test,
                                                  directory='train/',
                                                  x_col='filename',
                                                  y_col='label',
                                                  color_mode='rgb',
                                                  batch_size=batch_size,
                                                  validate_filenames=False,
                                                  shuffle=False,
                                                  target_size=(image_size, image_size))
