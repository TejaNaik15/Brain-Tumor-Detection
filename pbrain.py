from codecs import BOM32_BE
from ctypes import alignment
from unittest import result
from xml.dom.expatbuilder import parseString
import numpy as np
import pandas as pd
import pydicom as dicom
import os
import matplotlib.pyplot as plt
import cv2
import math
import numpy as np 
import matplotlib.pyplot as plt #for plotting things
import os
from tkinter import filedialog
from PIL import Image,ImageTk
from tkinter.filedialog import askopenfile
#print(os.listdir("../input"))
 
# Keras Libraries
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator, load_img
import tensorflow._api.v2.compat.v1 as tf 

import pandas as pd
import tflearn
from tflearn.layers.conv import conv_3d, max_pool_3d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from tkinter import *
from tkinter import messagebox,ttk
import tkinter as tk
from PIL import Image,ImageTk

cnn = Sequential()#Convolution 
class LCD_CNN:
    def __init__(self,root):
        self.root=root
        #window size
        self.root.geometry("1006x500+0+0")
        self.root.resizable(False, False)
        self.root.title("Brain Tumor Detection")

        img4=Image.open(r"xray/train/Tumor/Y55.jpg")
        img4=img4.resize((1006,500),Image.ANTIALIAS)
        self.photoimg4=ImageTk.PhotoImage(img4)

        bg_img=Label(self.root,image=self.photoimg4)
        bg_img.place(x=0,y=50,width=1006,height=500)

        # title Label
        title_lbl=Label(text="Brain Tumor Detection",font=("Bradley Hand ITC",30,"bold"),bg="black",fg="white",)
        title_lbl.place(x=0,y=0,width=1006,height=50)

        #photos Button
        # img10=Image.open(r"Images\opencv_face_reco_more_data.jpg")
        # img10=img10.resize((180,140),Image.ANTIALIAS)
        # self.photoimg10=ImageTk.PhotoImage(img10)

        #button 1
        self.b1=Button(text="Import Data",cursor="hand2",command=self.import_data,font=("Times New Roman",15,"bold"),bg="white",fg="black")
        self.b1.place(x=80,y=130,width=180,height=30)

        #button 3
        self.b3=Button(text="Train Data",cursor="hand2",command=self.train_data,font=("Times New Roman",15,"bold"),bg="white",fg="black")
        self.b3.place(x=80,y=180,width=180,height=30)
        self.b3["state"] = "disabled"
        self.b3.config(cursor="arrow")
        
        #button 4
        self.b4=Button(text="Test Data",cursor="hand2",command=self.test_data,font=("Times New Roman",15,"bold"),bg="white",fg="black")
        self.b4.place(x=80,y=230,width=180,height=30)
        self.b4["state"] = "disabled"
        self.b4.config(cursor="arrow")

    def import_data(self):
        ##Data directory
        self.dataDirectory = 'xray/train/'
        self.TumorPatients = os.listdir(self.dataDirectory)
        ##Setting x*y size to 50
        self.size = 10
        ## Setting z-dimension (number of slices to 20)
        self.NoSlices = 5
        messagebox.showinfo("Import Data" , "Data Imported Successfully!")

        self.b1["state"] = "disabled"
        self.b1.config(cursor="arrow")
        self.b3["state"] = "normal"
        self.b3.config(cursor="hand2")

    def train_data(self):
        cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))

        #Pooling
        cnn.add(MaxPooling2D(pool_size = (2, 2)))

        # 2nd Convolution
        cnn.add(Conv2D(32, (3, 3), activation="relu"))

        # 2nd Pooling layer
        cnn.add(MaxPooling2D(pool_size = (2, 2)))

        # 3nd Convolution
        cnn.add(Conv2D(32, (3, 3), activation="relu"))

        # 3nd Pooling layer
        cnn.add(MaxPooling2D(pool_size = (2, 2)))

        # Flatten the layer
        cnn.add(Flatten())

        # Fully Connected Layers
        cnn.add(Dense(activation = 'relu', units = 128))
        cnn.add(Dense(activation = 'sigmoid', units = 1))

        # Compile the Neural network
        cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        num_of_test_samples = 200
        batch_size = 32
        # Fitting the CNN to the images
        # The function ImageDataGenerator augments your image by iterating through image as your CNN is getting ready to process that image

        train_datagen = ImageDataGenerator(rescale = 1./255,
                                        shear_range = 0.2,
                                        zoom_range = 0.2,
                                        horizontal_flip = True)

        test_datagen = ImageDataGenerator(rescale = 1./255)  #Image normalization.

        training_set = train_datagen.flow_from_directory('xray/train',
                                                        target_size = (64, 64),
                                                        batch_size = 32,
                                                        class_mode = 'binary')

        validation_generator = test_datagen.flow_from_directory('xray/val/',
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary')

        test_set = test_datagen.flow_from_directory('xray/test',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'binary')
        cnn_model = cnn.fit_generator(training_set,
                         steps_per_epoch = 8,
                         epochs = 9,
                         validation_data = validation_generator,
                         validation_steps = 20)
        test_accu = cnn.evaluate_generator(test_set,steps=20)
        print('The testing accuracy is :',test_accu[1]*100, '%')
        messagebox.showinfo("ACCURACY" ,test_accu[1]*100)
        messagebox.showinfo("Train Data" , "Model Trained Successfully!")

        self.b3["state"] = "disabled"
        self.b3.config(cursor="arrow")
        self.b4["state"] = "normal"
        self.b4.config(cursor="hand2")

    def test_data(self):
        f_types = [('Jpg Files', '*.jpg')]
        filename = filedialog.askopenfilename(filetypes=f_types)
        img = ImageTk.PhotoImage(file=filename)


        from keras.preprocessing import image
        import matplotlib.image as mpimg
        
        img = mpimg.imread(filename)
        plt.imshow(img)
        plt.show()
        img = image.load_img(filename, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        classes = cnn.predict(x)
        print(classes)
        def ans():
            if classes>0.5:
                return("Tumor")
            else:
                return("Normal")

        messagebox.showinfo("Test Data" , ans()) 

if __name__ == "__main__":
        root=Tk()
        obj=LCD_CNN(root)
        root.mainloop()

