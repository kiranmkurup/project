
import os
import cv2
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.applications import InceptionV3
import pickle

from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import keras.backend as K
import os
import cv2 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
import pandas as pd
import numpy
import pickle
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import numpy

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Dropout, Activation   
from imblearn.over_sampling import SMOTE
f=open('newdata_image.pkl','rb')
datalist=pickle.load(f)
f.close()

g=open('newlabel_image.pkl','rb')
labellist=pickle.load(g)
g.close()

# print(datalist)
print(datalist[0].shape)
# cv2.imshow("image",datalist[0])
# cv2.waitKey(0)
# traindata=[]
# for i in datalist:
#     img=cv2.resize(i,(128,128))
#     img1=[img,img]
#     img1=np.array(img1)
#     traindata.append(img1)
# f=open('newdata_image1.pkl','wb')
# pickle.dump(traindata,f)
# f.close()
f=open('newdata_image1.pkl','rb')
datalist=pickle.load(f)
f.close()
traindata=np.array(datalist)
print(traindata.shape)
lb=LabelEncoder()
ydata = np_utils.to_categorical(lb.fit_transform(labellist))
print(ydata)
print(traindata.shape)

X_train, X_test, y_train, y_test = train_test_split(traindata, ydata, test_size=0.25, random_state=42)
from model import main1

model=main1((2,128,128,3))
checkpointer = ModelCheckpoint(filepath='dynamicmodel.h5', verbose=1, save_best_only=True ,monitor='val_accuracy')


model.fit(X_train, y_train, epochs=70,
        shuffle=True,
        batch_size=16, validation_data=(X_test, y_test),
        callbacks=[checkpointer], verbose=1)

model_json = model.to_json()
with open("dynamicmodel.json", "w") as json_file:
    json_file.write(model_json)   



