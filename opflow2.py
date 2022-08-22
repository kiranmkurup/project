

import cv2
import random
import os
import pickle
import numpy as np
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import Input
from keras.models import Model
from keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
base_model = InceptionResNetV2(weights='imagenet', include_top=True)
model = Model(inputs=base_model.input,outputs=base_model.get_layer('avg_pool').output)
def extract(image_path):
    

    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Get the prediction.
    features = model.predict(x)    
    features = features[0]
    return features

datalist=[]
labellist=[]
def getoptical(impath,sfname):
    print(sfname)
    try:
        imglist = os.listdir(impath)
        # print(imglist)
        lstlen=len(imglist)
        numfrm=int(lstlen/5)
        pval=0
      
        start=0
        end=numfrm
        featlist=[]
        samplelist=[]
        while(start<lstlen):   
            framelist=imglist[start:end]
            samplelist.append(framelist)
            # print("======================")
            start=start+numfrm
            end=end+numfrm
       
        # print("sample list==>",samplelist)

        rlist=[]
        for i in samplelist:
            # print("sample",i)
            frame1 = cv2.imread(impath+i[0])
            width = int(frame1.shape[1])
            height = int(frame1.shape[0])
            frame1 = cv2.resize(frame1, (width, height), interpolation = cv2.INTER_AREA)

            prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
            hsv = np.zeros_like(frame1)
            hsv[...,1] = 255
            frlist=[]
            for j in i:
                
                frame2 = cv2.imread(impath+j)

                width = int(frame2.shape[1])
                height = int(frame2.shape[0])
                frame2 = cv2.resize(frame2, (width, height), interpolation = cv2.INTER_AREA)

                next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                # print("flow", flow)
                
                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                hsv[...,0] = ang*180/np.pi/2
                hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
                bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
                dst = cv2.addWeighted(frame2, 0.2, bgr, 0.7, 0)
                frlist.append(dst)
                # cv2.imshow('opflow',dst)
                # cv2.waitKey(0)
            rlist.append(frlist)

        # print(len(rlist[0]))
        for k in rlist:
            klen=len(k)
            rnum = random.randint(0,klen-1)
            selframe=k[rnum]
            # cv2.imwrite("dst.png",selframe)
            # features = extract("dst.png")
            # print(features)
            datalist.append(selframe)
            labellist.append(sfname)
            # cv2.imshow('opflow',selframe)
            # cv2.waitKey(0)

        print("======================")
    except Exception as e:
        print("error",e)

# impath="dataset3/Pushing Hand Away/14/"
# getoptical(impath,'Pushing Hand Away')

path='dataset1'
subfolders= [f for f in os.scandir(path) if f.is_dir()]

# datalist=[]
# labelslist=[]

for sf in subfolders:
        imgs1=[f for f in os.scandir(sf.path)]
        for img in imgs1:
            path1=path+'/'+sf.name+'/'+img.name+'/'
            # print(path1)
            # print(sf.name)
            getoptical(path1,sf.name)
            

            

f=open('newdata_image_no.pkl','wb')
pickle.dump(datalist,f)
f.close()

g=open('newlabel_image_no.pkl','wb')
pickle.dump(labellist,g)
g.close()