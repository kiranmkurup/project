from keras.models import model_from_json

with open('dynamicmodel.json', "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights('dynamicmodel.h5')

import cv2
import os,shutil
import random
import numpy as np
def getoptical(impath):
    datalist=[]
    try:
        imglist = os.listdir(impath)
        print(imglist)
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
     
        
        return selframe
        

        print("======================")
    except Exception as e:
        print("error",e)


def getvideo(vidpath):
    try: 
        cap = cv2.VideoCapture(vidpath)
        i = 0
        length = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
        print( length )
        while(cap.isOpened()):
            print(i)
            ret, frame = cap.read()
            cv2.imwrite("frames/frame"+str(i)+".jpg",frame)
            cv2.imshow("Image", frame)
            cv2.waitKey(1)
            
            i+=1
            # else:
            #   i+=1
            
        
        cap.release()
        cv2.destroyAllWindows()
    except:
        pass


dir = 'frames/'
for files in os.listdir(dir):
    path = os.path.join(dir, files)
    try:
        shutil.rmtree(path)
    except OSError:
        os.remove(path)
getvideo('test/stop5.avi')
opfeat=getoptical("frames/")
# print(opfeat)
# cv2.imshow("opfeat",opfeat)
# cv2.waitKey(0)

img=cv2.resize(opfeat,(128,128))
img1=[img,img]
img1=np.array(img1)
print(img1.shape)
xtest=np.expand_dims(img1, axis=0)
pred=loaded_model.predict(xtest)
print(pred)
print(np.argmax(pred[0]))



