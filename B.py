from tensorflow.keras.models import load_model
import cv2
import numpy as np
import tensorflow as tf

# We need not to create and train the model again
# No need to train the model, its a pre defined trained model
model = load_model("C:/Users/Param Prashar/Desktop/Soft Computing Project/model.h5")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Testing the Normal Image
# image = cv2.imread("C:/Users/Param Prashar/Desktop/Soft Computing Project/covid19dataset/test/normal/NORMAL2-IM-1385-0001.jpeg")    # 1
image = cv2.imread("C:/Users/Param Prashar/Desktop/Soft Computing Project/covid19dataset/test/covid/nejmoa2001191_f3-PA.jpeg")    # 0
image = cv2.resize(image, (64, 64))
image = np.reshape(image, [1, 64, 64, 3])

classes = model.predict_classes(image)
label = ["COVID-19 INFECTED", "NORMAL"]
print(classes)
print(label[classes[0][0]])
# tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=False, show_layer_names=True)

"""
    Upon Finishing the Training we have Loss and Accuracy
    So, after loading the model validate if it has the same Loss and Accuracy as we trained in for

    Try loading the model only and not h5 format

"""
