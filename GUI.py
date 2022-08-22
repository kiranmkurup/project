from tkinter import *

from PIL import Image, ImageTk  
import cv2
from tkinter.filedialog import askopenfile
import numpy as np
import tensorflow as tf
# tf.compat.v1.keras.backend
from tensorflow.keras.models import model_from_json
import cv2
import os,shutil
import numpy as np
import random
pathname=""

class_names = ["Pushing Hand Away", "Stop","Swiping Right","Swiping Up"] 

with open('dynamicmodel.json', "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights('dynamicmodel.h5')

print("Model loaded....")

dir = 'frames/'
for files in os.listdir(dir):
    path = os.path.join(dir, files)
    try:
        shutil.rmtree(path)
    except OSError:
        os.remove(path)

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

        cv2.imwrite("img4b.jpg",bgr)
        # cv2.imshow("bgr",bgr)
        
        # cv2.waitKey(0)
        img = cv2.imread('img4b.jpg',0) #read img as b/w as an numpy array
        unique, counts = np.unique(img, return_counts=True)
        mapColorCounts = dict(zip(unique, counts))
        ccount=mapColorCounts[0]
        print(mapColorCounts[0])
        # image =bgr
        # print("count non zero==>",cv2.countNonZero(image))
        # if cv2.countNonZero(image) == 0:
        #     print ("Image is black")
        # else:
        #     print ("Colored image")
        # print(len(rlist[0]))
        for k in rlist:
            klen=len(k)
            rnum = random.randint(0,klen-1)
            selframe=k[rnum]
     
        if(ccount>850000):
            return "no motion"
        else:
            return selframe
        
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
            # cv2.imshow("Image", frame)
            # cv2.waitKey(1)
            
            i+=1
            # else:
            #   i+=1
            
        
        cap.release()
        cv2.destroyAllWindows()
    except:
        pass



def pp(a):
    global mylist
    mylist.insert(END, a)



def predict(val):
    global mylist
    print("path name-->",val)
    getvideo(val)
    opfeat=getoptical("frames/")
    if(opfeat=="no motion"):
        prediction="No motion"
    else:
    # print(opfeat)
        img=cv2.resize(opfeat,(128,128))
        img1=[img,img]
        img1=np.array(img1)
        print(img1.shape)
        xtest=np.expand_dims(img1, axis=0)
        pred=loaded_model.predict(xtest)
        print(pred)
        print(np.argmax(pred[0]))
        y_pred_class=np.argmax(pred[0])
        prediction = class_names[y_pred_class]
        prob=(pred[0][y_pred_class])*100
        print("prob==>",prob)
        root.after(3100, lambda :shrslt.config(text=prediction,fg="red"))
    print(prediction)
    root.after(500, lambda : pp("Preprocessing Started "))
    root.after(2000, lambda : pp("Features Extraction"))
    root.after(2300, lambda : pp("Model Loaded"))
    root.after(2500, lambda : pp("Prediction using loaded model"))
    root.after(2800, lambda : pp("Result: "+prediction))
    root.after(3000, lambda : pp("============================"))
    
        
    
        
def browseim():
    global cimg,shrslt,E1,lblinfo3,pathname
    path = askopenfile()
    n=path.name 
    pathname=n
    print(n)
    nlist=n.split("/")
    vpth=nlist[-1]
    print(vpth)
    lblinfo3.config(text="Selected video:"+vpth,fg="red")
    
    
def userHome():
    global root, mylist,shrslt,E1,lblinfo3,pathname
    root = Tk()
    root.geometry("1200x700+0+0")
    root.title("Home Page")

    image = Image.open("hand2.jpg")
    image = image.resize((1200, 700), Image.ANTIALIAS) 
    pic = ImageTk.PhotoImage(image)
    lbl_reg=Label(root,image=pic,anchor=CENTER)
    lbl_reg.place(x=0,y=0)
  
    #-----------------INFO TOP------------
    lblinfo = Label(root, font=( 'aria' ,20, 'bold' ),text="Dynamic hand gesture recognition system",fg="white",bg="#000955",bd=10,anchor='w')
    lblinfo.place(x=350,y=50)
 
    lblinfo3 = Label(root, font=( 'aria' ,15 ),text="",fg="#000955",anchor='w')
    lblinfo3.place(x=480,y=360)
    
    mylist = Listbox(root,width=60, height=20,bg="white")

    mylist.place( x = 50, y = 300 )
    mylist.insert(END,"Process")
    mylist.insert(END,"===================================")
    btntrn=Button(root,padx=50,pady=2, bd=4 ,fg="white",font=('ariel' ,16,'bold'),width=10, text="Browse a video to test", bg="red",command=lambda:browseim())
    btntrn.place(x=480, y=300)


    btnhlp=Button(root,padx=80,pady=8, bd=6 ,fg="white",font=('ariel' ,10,'bold'),width=7, text="Find gesture", bg="green",command=lambda:predict(pathname))
    btnhlp.place(x=480, y=400)

    rslt = Label(root, font=( 'aria' ,20, ),text="RESULT :",fg="black",bg="white",anchor=W)
    rslt.place(x=450,y=480)
    shrslt = Label(root, font=( 'aria' ,20, ),text="",fg="blue",bg="white",anchor=W)
    shrslt.place(x=590,y=480)


    def qexit():
        root.destroy()
     

    root.mainloop()


userHome()