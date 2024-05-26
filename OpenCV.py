#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install opencv-contrib-python


# In[1]:


pip install caer


# In[2]:


import cv2 as cv


# In[2]:


#reading Photos

img = cv.imread('cat.jpg')

cv.imshow('Cat', img)

cv.waitKey(0)


# In[3]:


#reading videos

capture = cv.VideoCapture('dog.mp4')

while True:
    isTrue, frame = capture.read()
    
    cv.imshow('Video', frame)
    
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
        
capture.release()
capture.destroyAllWindows()


# In[4]:


#rescaling and resizing

def rescaleFrame(frame, scale = 0.75):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    
    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)


capture = cv.VideoCapture('dog.mp4')

while True:
    isTrue, frame = capture.read()
    
    frame_resized = rescaleFrame(frame)
    
    cv.imshow('Video', frame)
    cv.imshow('Video Resized', frame_resized)
    
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
        
capture.release()


# In[2]:


#blank

import numpy as np

blank = np.zeros((500,500), dtype = 'uint8')

cv.imshow('Blank',blank)

img = cv.imread('cat.jpg')

cv.imshow('Cat', img)

cv.waitKey(0)


# In[2]:


#draw a rectangle

import numpy as np

blank = np.zeros((500,500,3), dtype = 'uint8')

cv.rectangle(blank, (0,0), (250,250), (0,255,0), thickness=2)

cv.imshow('Rectangle', blank)

cv.waitKey(0)


# In[3]:


#draw green box in a blank

import numpy as np

blank = np.zeros((500,500,3), dtype = 'uint8')

#cv.imshow('Blank',blank)

blank[200:300,300:400] = 0,255,0

cv.imshow('Green', blank)

cv.waitKey(0)


# In[4]:


#draw a rectangle

import numpy as np

blank = np.zeros((500,500,3), dtype = 'uint8')

cv.rectangle(blank, (0,0), (250,500), (0,255,0), thickness=cv.FILLED)

cv.imshow('Rectangle', blank)

cv.waitKey(0)


# In[6]:


#put some text

import numpy as np

blank = np.zeros((500,500,3), dtype = 'uint8')

cv.putText(blank, 'Hello, my name is Ananya', (0,255), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0), 2)

cv.imshow('Text', blank)



cv.waitKey(0)


# In[2]:


#converting image to grayscale

img = cv.imread('cat.jpg')

#cv.imshow('Cat', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)


cv.waitKey(0)


# In[7]:


img = cv.imread('cat.jpg')

cv.imshow('Cat', img)

#blurring an image

blur = cv.GaussianBlur(img,(7,7), cv.BORDER_DEFAULT) #increase kernel size to increase blur
cv.imshow('Blur', blur)

canny = cv.Canny(blur, 125, 175)

cv.imshow('Canny', canny)


cv.waitKey(0)


# In[3]:


#canny edge detector
#Edge cascade
img = cv.imread('cat.jpg')

#cv.imshow('Cat', img)

canny = cv.Canny(img, 125, 175)

#cv.imshow('Canny', canny)

#dilating the image

dilated = cv.dilate(canny, (7,7), iterations = 1) #iterations increases edge thickness
cv.imshow('Dilated', dilated)


#eroding
erode = cv.erode(dilated, (3,3), iterations=1)
cv.imshow('Erode', erode)

cv.waitKey(0)



# In[3]:


img = cv.imread('cat.jpg')

cv.imshow('Cat', img)

#resizing

resized = cv.resize(img, (500,500), interpolation = cv.INTER_CUBIC)
cv.imshow('Resized', resized)

#cropping

cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)



cv.waitKey(0)


# In[3]:


import numpy as np


# In[4]:


img = cv.imread('cat.jpg')

cv.imshow('Cat', img)

def translate(img,x,y):
    transMat = np.float32([[1,0,x], [0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

translated = translate(img,100,100)

cv.imshow('Translate', translated)

cv.waitKey(0)


# In[6]:


img = cv.imread('cat.jpg')

#cv.imshow('Cat', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv)

lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB', lab)


cv.waitKey(0)


# In[3]:


import matplotlib.pyplot as plt

img = cv.imread('cat.jpg')

cv.imshow('Cat', img)

plt.imshow(img)
plt.show()


# In[2]:


import cv2 as cv
import numpy as np

blank = np.zeros((400,400), dtype='uint8')

rectangle = cv.rectangle(blank.copy(), (30,30),(370,370),255, -1)
circle = cv.circle(blank.copy(),(200,200), 200, 255, -1)

cv.imshow('Rectangle', rectangle)
cv.imshow('Circle', circle)

#bitwise AND

bitwise_and = cv.bitwise_and(rectangle, circle)
cv.imshow('Bitwise AND', bitwise_and)

cv.waitKey(0)


# In[8]:


import cv2 as cv
import numpy as np

img = cv.imread('cat.jpg')

cv.imshow('Cat', img)

blank =  np.zeros(img.shape[:2], dtype = 'uint8')
cv.imshow('Blank', blank)


mask = cv.circle(blank, (img.shape[1]//2 + 90, img.shape[0]//2 - 100), 100, 255, -1)

#cv.imshow('Mask', mask)

masked = cv.bitwise_and(img,img, mask=mask)
cv.imshow('Masked Image', masked)

cv.waitKey(0)


# In[12]:


import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt

img = cv.imread('cat.jpg')

cv.imshow('Cat', img)




gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)



gray_hist = cv.calcHist([gray], [0], None, [256], [0,256])

plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show()




cv.waitKey(0)


# In[13]:


import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('cat.jpg')

cv.imshow('Cat', img)

blank = np.zeros(img.shape[:2], dtype='uint8')
mask = cv.circle(blank, (img.shape[1]//2,img.shape[0]//2), 100, 255, -1)

masked = cv.bitwise_and(img,img,mask=mask)
cv.imshow('Mask', masked)

plt.figure()
plt.title('Colour Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
colors = ('b', 'g', 'r')
for i,col in enumerate(colors):
    hist = cv.calcHist([img], [i], mask, [256], [0,256])
    plt.plot(hist, color=col)
    plt.xlim([0,256])

plt.show()

cv.waitKey(0)


# In[1]:


import cv2 as cv
import numpy as np

img = cv.imread('cat.jpg')

#cv.imshow('Cat', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)


# In[ ]:




