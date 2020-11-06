import cv2
import time
import numpy as np

#Read Image_______________________________________________________________________________
Ti = time.time() #Initial time.
num = 7000
hand_name = "G:/My Drive/Semestres/2019/2019-10/Vision Artificial/Tareas/Tarea 3/Databases/Hands/Hand_"+str(num).zfill(7)+".jpg" 
#For hand_name please use your own directory.
#hand_name = 'Test.jpg'
I = cv2.imread(hand_name)
I2 = cv2.convertScaleAbs(I,0,1) #Raise image's contrast.
#I = cv2.resize(I,(1920,1080),fx=0,fy=0,interpolation=cv2.INTER_CUBIC) #Resize image.

# Compare pixel to pixel with skin color range (predefined)_______________________________
skin_low = np.array([0,48,80], dtype='uint8') #Lower limit for skin detection
skin_high = np.array([20,255,255], dtype='uint8') #Upper limit for skin detection
#Everything outside these bounds will be considered as not-skin and will be painted black.
I_HSV = cv2.cvtColor(I,cv2.COLOR_BGR2HSV) #Change the image's colorspace to HSV.
hgt,wdt,chn = I_HSV.shape #Get the image's dimensions.
mask = cv2.inRange(I_HSV,skin_low,skin_high) #If the values are within the boundaries,
#pixel will be 255; otherwise it will be 0. In other words, paint black whatever is not skin.
skin = cv2.bitwise_and(I,I,mask=mask) #This changes the 255 for the original pixel value,
#to preserve the original image's color in RGB.

#Detection of True Positives to Make ROC Curve____________________________________________
tp = 0 #True Positives.
tn = 0 #True Negatives.
fp = 0 #False Positives.
fn = 0 #False Negatives.
for h in range(hgt):
    for w in range(wdt):
        if I2[h,w][2] < 215 and I2[h,w][0] < 160: #If I2 pixels' values are within these values,
            #then the pixel is skin.
            if skin[h,w][2] != 0:
                tp += 1 #It means it found a true positive.
            else:
                fn += 1 #It means it found a false positive.
        else:
            if skin[h,w][2] != 0:
                fp += 1 #It means it found a false negative.
            else:
                tn += 1 #It means it found a true negative.

print(hand_name[-16:-4])                
print("True Positive Rate: "+str(round(100*tp/(tp+fp),2))+"%")
print("False Positive Rate: "+str(round(100*fp/(fn+tn),2))+"%")

#Show Images______________________________________________________________________________
subplt = cv2.hconcat([I2,skin])
cv2.namedWindow('Original v/s Detector',cv2.WINDOW_NORMAL)
cv2.imshow('Original v/s Detector',subplt)
cv2.imwrite('Comparison.jpg',subplt)


#Execution time and end program___________________________________________________________
Tf = round(time.time()-Ti,3) #Final time to get execution time.
print("Execution Time: ",Tf," seconds.")
