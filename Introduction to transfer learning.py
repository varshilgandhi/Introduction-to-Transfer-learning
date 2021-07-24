# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 22:18:14 2021

@author: abc
"""

from keras.applications.vgg16 import VGG16

#Define model
model = VGG16()

#Check how it look like
model.summary()

#Load our image
from keras.preprocessing.image import load_img
#image = load_img("cab.jpg", target_size = (224, 224))
image = load_img("aeroplane.jpg", target_size = (224, 224))

#Convert image into numpy array format
from keras.preprocessing.image import img_to_array
image = img_to_array(image)

#Give the shape to our image 
import numpy as np
image = np.expand_dims(image, axis=0)

# To get better results we need preprocess our results
from keras.applications.vgg16 import preprocess_input
image = preprocess_input(image)

#Predict our model
pred = model.predict(image)

#Print Our prediction
from tensorflow.keras.applications.mobilenet import decode_predictions
pred_classes = decode_predictions(pred, top=5)
for i in pred_classes[0] :
    print(i)
    
    
    
