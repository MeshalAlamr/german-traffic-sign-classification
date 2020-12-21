import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import matplotlib.pyplot as plt


EPOCHS = 10 # number of epochs
WIDTH = 30  # width of image
HEIGHT = 30 # height of image
CATEGORIES = 43 # number of categories, in the data they're (0 to 42)
MODEL = 3 # model number to be selected (from 1 to 3)


def main():
    
    # check command-line arguments, if they're not correct exit and report
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python sign.py data_directory [modelname.h5]. saving the model is optional.")
    
    # set the specified data directory
    data_dir = sys.argv[1]
    
    # create lists for images and labels
    images = []
    labels = []
    
    # define the folder path
    folder_path = os.path.join(".", f"{data_dir}")
    
    # store the found labels
    categories = [f for f in os.listdir(folder_path)]

    # iterate over all categories
    for index, category in enumerate(categories):
        
        # get the category path
        category_path = os.path.join(folder_path, f"{category}")
        
        # store the found files in the category 
        files = os.listdir(category_path)
        
        # iterate over all files in the category
        for file in files:
            
            # get the file path            
            file_path = os.path.join(folder_path, f"{category}", f"{file}")
            
            # read each image and resize it
            img = cv2.imread(file_path) 
            img = cv2.resize(img,(WIDTH, HEIGHT))     
            
            # append the lables and images
            labels.append(category)
            images.append(img)
            
        # print current progress
        print(f"Category {category} loaded succesffuly.\nTotal categories loaded: {index+1}/{CATEGORIES}. ")
        
    # print progress completion
    print(f"Succeffuly loaded all {CATEGORIES} categories.")

    # converts labels into one hot encoding
    labels = to_categorical(labels)
    
    # split the data into training and testing
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=0.4)
    
    # select wanted model of neural network (from 1 to 3)
    model = select_model(MODEL)

    # fit model on training data
    history = model.fit(x_train, y_train, batch_size=32, epochs=EPOCHS, validation_data=(x_test, y_test))
    
    # evaluate the performance of the neural network
    model.evaluate(x_test,  y_test, verbose=2)

    # save the model to file if specified
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")

    # plot accuracy
    plt.figure(0)
    plt.plot(history.history["accuracy"], label="training accuracy")
    plt.plot(history.history["val_accuracy"], label="val accuracy")
    plt.title("Accuracy Plot")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.xticks(range(0, EPOCHS))
    plt.grid
    plt.show()

    # plot loss
    plt.figure(1)
    plt.plot(history.history["loss"], label="training loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.title("Loss Plot")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks(range(0, EPOCHS))
    plt.grid
    plt.show()


def select_model(x):
    
    # initiate the sequential model
    model = tf.keras.models.Sequential()
    
    if x == 1:
        # Model 1
        model.add(Conv2D(32, (3,3), activation = "relu", input_shape = (WIDTH, HEIGHT,3)))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Conv2D(64, (3,3), activation = "relu"))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Conv2D(128, (3,3), activation = "relu"))
        model.add(MaxPool2D(pool_size=(2,2)))    
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(CATEGORIES, activation="softmax"))
        
    elif x == 2:
        # Model 2
        model.add(Conv2D(32, (5,5), activation = "relu", input_shape = (WIDTH, HEIGHT,3)))
        model.add(Conv2D(filters=32, kernel_size=(5,5), activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Flatten())
        model.add(Dropout(rate=0.5))
        model.add(Dense(CATEGORIES, activation="softmax"))
        
    elif x == 3:
        # Model 3
        model.add(Conv2D(32, (5,5), activation = "relu", input_shape = (WIDTH, HEIGHT,3)))
        model.add(Conv2D(filters=32, kernel_size=(5,5), activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), padding="same"))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), padding="same"))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=128, kernel_size=(2, 2), activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(2, 2), activation="relu"))
        model.add(MaxPool2D(pool_size=(2, 2), padding="same"))
        model.add(Dropout(rate=0.25))
        model.add(Flatten())        
        model.add(Dropout(rate=0.5))
        model.add(Dense(CATEGORIES, activation="softmax"))
    
    else:
        sys.exit("Invalid model selection")
        
    # compile the model with the default optimizer
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    # return a compiled neural network based on the selected model
    return model



main()