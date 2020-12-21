import sys
import cv2
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


WIDTH = 30  # width of image
HEIGHT = 30 # height of image


def main():
    
    # check command-line arguments, if they're not correct exit and report
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python sign_test.py model_directory image_directory.")
     
    # load the model
    model = load_model(sys.argv[1])
    
    # classes of traffic signs from the GTSRB dataset
    classes = { 
                0:'Speed limit (20km/h)',
                1:'Speed limit (30km/h)', 
                2:'Speed limit (50km/h)', 
                3:'Speed limit (60km/h)', 
                4:'Speed limit (70km/h)', 
                5:'Speed limit (80km/h)', 
                6:'End of speed limit (80km/h)', 
                7:'Speed limit (100km/h)', 
                8:'Speed limit (120km/h)', 
                9:'No passing', 
                10:'No passing veh over 3.5 tons', 
                11:'Right-of-way at intersection', 
                12:'Priority road', 
                13:'Yield', 
                14:'Stop', 
                15:'No vehicles', 
                16:'Veh > 3.5 tons prohibited', 
                17:'No entry', 
                18:'General caution', 
                19:'Dangerous curve left', 
                20:'Dangerous curve right', 
                21:'Double curve', 
                22:'Bumpy road', 
                23:'Slippery road', 
                24:'Road narrows on the right', 
                25:'Road work', 
                26:'Traffic signals', 
                27:'Pedestrians', 
                28:'Children crossing', 
                29:'Bicycles crossing', 
                30:'Beware of ice/snow',
                31:'Wild animals crossing', 
                32:'End speed + passing limits', 
                33:'Turn right ahead', 
                34:'Turn left ahead', 
                35:'Ahead only', 
                36:'Go straight or right', 
                37:'Go straight or left', 
                38:'Keep right', 
                39:'Keep left', 
                40:'Roundabout mandatory', 
                41:'End of no passing', 
                42:'End no passing veh > 3.5 tons' 
                }
    
    
    data=[]    

    # read the image and resize it    
    img = cv2.imread(sys.argv[2]) 
    img = cv2.resize(img,(WIDTH, HEIGHT))
    
    # predict
    data.append(np.array(img))
    test = np.array(data)
    pred = model.predict_classes(test)
    
    # predicted category
    for i in pred: text = [str(i)] 
    class_key = int("".join(text)) 
    
    # plot the image with the prediction as the title
    plt.figure(0)
    plt.imshow(img)
    plt.title(f"Predicted: {classes[class_key]}")
    plt.show()

main() 
