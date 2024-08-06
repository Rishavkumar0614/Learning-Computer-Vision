import os
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
segmentor = SelfiSegmentation()
listImg = os.listdir("images")
imgList = []
for imgPath in listImg:
    img = cv2.imread(os.path.join("images", imgPath))
    img = cv2.resize(img, (640, 480))
    imgList.append(img)

i = 0

while True:
    success, img = cap.read()
    imgOut = segmentor.removeBG(img, imgList[i], cutThreshold = 0.5)
    
    
    imgStacked = cvzone.stackImages([img, imgOut], 2, 1)
    if success:
        cv2.imshow("Image", imgStacked)
        key = cv2.waitKey(1)
        if(key == ord('c')):
            break
        elif(key == ord('a') and i > 0):
            i = i - 1
        elif(key == ord('d') and i < (len(listImg) - 1)):
            i = i + 1
        
cap.release()
cv2.destroyAllWindows()