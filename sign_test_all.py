import sys
import cv2
import pandas as pd
from keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score


WIDTH = 30  # width of image
HEIGHT = 30 # height of image


def main():
    
    # check command-line arguments, if they're not correct exit and report
    if len(sys.argv) != 2:
        sys.exit("Usage: python sign_test_all.py model_directory")
     
    # load the model
    model = load_model(sys.argv[1])
    
    
    images = []
    labels = []
    
    # load the Test.csv file and assign labels and images accordingly
    y_test = pd.read_csv('Test.csv')
    labels = y_test["ClassId"].values
    images = y_test["Path"].values
    
    data=[]
    
    # go through each image in the file, read it, reszie it and append it
    for image in images:
        img = cv2.imread(image) 
        img = cv2.resize(img,(WIDTH, HEIGHT))
        data.append(np.array(img))
    
    # predict
    x_test=np.array(data)
    pred = model.predict_classes(x_test)
    
    # print the accuracy
    print(accuracy_score(labels, pred))

main() 
