import cv2
import numpy as np

import os
# def draw_flow(img, flow, step=25):
#     # print(img.shape)
#     h, w = img.shape[:2]
#     y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
#     fx, fy = flow[y,x].T
#     lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
#     lines = np.int32(lines + 0.5)
#     # vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     vis = img
#     cv2.polylines(vis, lines, 0, (0, 255, 0), 2)
#     for (x1, y1), (x2, y2) in lines:
#         cv2.circle(vis, (x1, y1), 2, (0, 0, 255), -1)
#         # vis = cv2.arrowedLine(vis, (x1,y1), (x2,y2), (0,255,0), 2) 
#     return vis


# cap = cv2.VideoCapture(0)
# ret, frame1 = cap.read()
# width = int(frame1.shape[1] * 0.75)
# height = int(frame1.shape[0] * 0.75)
# frame1 = cv2.resize(frame1, (width, height), interpolation = cv2.INTER_AREA)

# prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
# hsv = np.zeros_like(frame1)
# print('hsv', hsv)
# hsv[...,1] = 255
# print('new_hsv', hsv)

# while(1):
#     ret, frame2 = cap.read()

#     if ret == True:
#         width = int(frame2.shape[1] * 0.75)
#         height = int(frame2.shape[0] * 0.75)
#         frame2 = cv2.resize(frame2, (width, height), interpolation = cv2.INTER_AREA)

#         next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
#         flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#         # print("flow", flow)
        
#         mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
#         hsv[...,0] = ang*180/np.pi/2
#         hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
#         bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)


#         cv2.imshow('Cars with background subtraction',bgr)
       
#         k = cv2.waitKey(30) & 0xff
#         if k == 27:
#             break
#         prvs = next
    
#     else:
#         break

# cap.release()
# cv2.destroyAllWindows()


def getoptical(impath):
    imglist = os.listdir(impath)
    print(imglist)
    frame1 = cv2.imread(impath+imglist[0])
    width = int(frame1.shape[1])
    height = int(frame1.shape[0])
    frame1 = cv2.resize(frame1, (width, height), interpolation = cv2.INTER_AREA)

    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    print('hsv', hsv)
    hsv[...,1] = 255
    print('new_hsv', hsv)
    for i in imglist:
        frame2 = cv2.imread(impath+i)

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


        cv2.imshow('opflow',bgr)
        cv2.waitKey(0)

impath="dataset3/Pushing Hand Away/14/"
getoptical(impath)