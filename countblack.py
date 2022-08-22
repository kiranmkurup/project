import cv2
import numpy as np

img = cv2.imread('img3b.jpg',0) #read img as b/w as an numpy array
unique, counts = np.unique(img, return_counts=True)
mapColorCounts = dict(zip(unique, counts))
print(mapColorCounts[0])
