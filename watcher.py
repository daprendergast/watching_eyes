# import the necessary packages
from typing import TYPE_CHECKING
import numpy as np
import cv2
import math
import random
 
# initialize the HOG descriptor/person detector

def in_circle(center_x, center_y, radius, x, y):
    dist = math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
    return dist <= radius

def gate(value, minimum,  maximum):
    return max(min(value,maximum), minimum)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# open webcam video stream
cap = cv2.VideoCapture(0)

# the output will be written to output.avi




output_width = 1650
output_height = 1280
scale = int(output_width/640)
speed = int(.0125*output_width)/2

left_center =  (int(output_width/5),int(output_height/2))
right_center =  (int(output_width/5*4),int(output_height/2))

left_pupil_location = left_center
right_pupil_location = right_center

target = left_center
directed = False
attention = 20
iterations = 0
iris_size = 207*scale
oldframe = None
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (output_width,output_height))

while(True):
    iterations +=1
 
    if iterations > attention:
        print("Bored, looking elsewhere")
        directed = False
        iterations = 0
    # Capture frame-by-frame
    ret, frame = cap.read()

    # resizing for faster detection
    frame = cv2.resize(frame,  (640,480))
    frame = cv2.flip(frame,1)
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # detect people in the image
    # returns the bounding boxes for the detected objects
    #boxes = face_cascade.detectMultiScale(gray, 1.1, 4)
    boxes, weights = hog.detectMultiScale(gray, winStride=(8,8) )
    faceBoxes = []
    if oldframe is not None:
        oldframe = cv2.GaussianBlur(oldframe, (21,21),0)        
        newframe = cv2.GaussianBlur(gray, (21,21),0)
        subtraction = cv2.absdiff(oldframe, newframe)
        threshold = cv2.threshold(subtraction, 25,255, cv2.THRESH_BINARY)[1]
        threshold = cv2.dilate(threshold, None, iterations =2)
        contouring = threshold.copy()
        outlines, hierarchy = cv2.findContours(contouring, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)       
        a =0
        for c in outlines:            
            if cv2.contourArea(c) > a:
                a = cv2.contourArea(c)
                x,y,w,h = cv2.boundingRect(c)
                faceBoxes = [(int(x*scale+w*scale/2),int(y*scale))]
   
   
    oldframe = gray
    
    frame = cv2.resize(frame,  (output_width,output_height))

    #boxes = np.array([[x+int(w/2), y+ h, x + w, y + h] for (x, y, w, h) in boxes])
    #faceBoxes = np.array([[x*scale, y*scale] for (x, y, w, h) in boxes])
    currX, currY = left_pupil_location
  
    if (len(faceBoxes)>=1):
        speed = int(.0125*output_width)
        target = faceBoxes[0]
        directed = True
        print(target)
        print("Spotted you!")
        iterations = 0
        iris_size +=int(6*scale)
    else:
        iris_size -=int(3*scale)
    iris_size = gate(iris_size, 21*scale ,int(.0546*output_width))

    if not(directed):
        #print("Looking at new target")
        speed = int(4*scale)
        target = (currX + random.randint(-speed*200, speed*200), currY + random.randint(-speed*300, speed*200))
        #print(target)
        directed = True
        iterations = 0
        



    x,y =target
        # display the detected boxes in the colour picture
        #cv2.rectangle(frame, (xA, yA), (xB, yB),
        #                  (0, 255, 0), 2)
    if (x > currX): newX = currX+speed            
    else: newX = currX-speed
        
    if (y > currY): newY = currY+speed
    else: newY = currY-speed
    
    newY = max(min(newY,int(output_height/2+50)),int(.28125*output_height))
    if (in_circle(int(output_width/5),int(output_height/2),int(.14*output_width),newX, newY)):
        left_pupil_location = (newX,newY)
        right_pupil_location = tuple(np.add(right_pupil_location , (newX-currX, newY-currY)))
        # print("...")
        
    #else:
        #directed = False    
    
   
    arc = np.array([[1,1],[newX, newY],[output_width,output_height]], np.int32)
    arc = arc.reshape((-1, 1, 2))

        


    cv2.rectangle(frame,(0,0), (output_width,output_height), (0,0,0,10), thickness = -1)
    
    cv2.circle(frame,left_center,207*scale,(150,150,255, 255),thickness=-1)    #pink
    cv2.circle(frame,left_center,int(.2109*output_width),(225,225,255, 255),thickness=-1)    #white

    cv2.circle(frame,right_center,207*scale,(150,150,255, 255),thickness=-1)    #pink
    cv2.circle(frame,right_center,int(.2109*output_width),(225,225,255, 255),thickness=-1)    #white

    if (newY < int(output_height/2)): 
        cv2.ellipse(frame, (int(output_width/2),(int(.833*output_height))), (int(.36*output_width), int(.833*output_height)-newY),
            0, 200, 340, (150,150,255, 255), thickness = 1)     
    else:
         cv2.ellipse(frame, (int(output_width/2),int(.166*output_height)), (int(.36*output_width), newY-int(.166*output_height)),
            0, 35, 155, (150,150,255, 255), thickness = 1)     


    cv2.circle(frame,left_pupil_location,int(.078125*output_width),(3,39,148,150),thickness=-1)     #Iris
    cv2.circle(frame,left_pupil_location,iris_size,1,thickness=-1)                  #pupil
                     #eyelid

    cv2.circle(frame,right_pupil_location,int(.078125*output_width),(3,39,148,150),thickness=-1)     #Iris
    cv2.circle(frame,right_pupil_location,iris_size,1,thickness=-1)          
    
    cv2.ellipse(frame, left_center, (int(.38*output_width), int(.242*output_width)),
           19, 0, 360, 1, thickness = int(.15*output_width))                   #pupil
    cv2.ellipse(frame, right_center, (int(.38*output_width), int(.242*output_width)),
           -19, 0, 360, 1, thickness = int(.15*output_width))                                #eyelid
    # Write the output video 
    out.write(frame.astype('uint8'))
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# and release the output
out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)
