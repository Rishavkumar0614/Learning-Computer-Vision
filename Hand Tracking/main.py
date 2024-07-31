import cv2 
import math
import mediapipe as mp 
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class handDetector():
     def __init__(self, mode = False, maxHands = 2, complexity = 1, detectionCon = 0.5, trackCon = 0.5):
          self.mode = mode
          self.trackCon = trackCon
          self.maxHands = maxHands
          self.complexity = complexity
          self.detectionCon = detectionCon
          self.mpHands = mp.solutions.hands
          self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon)
          self.mpDraw = mp.solutions.drawing_utils
          devices = AudioUtilities.GetSpeakers()
          interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
          self.volume = cast(interface, POINTER(IAudioEndpointVolume))
     
     def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)  
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                index_finger_tip = handLms.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                x1 = index_finger_tip.x
                y1 = index_finger_tip.y
                thumb_finger_tip = handLms.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
                x2 = thumb_finger_tip.x
                y2 = thumb_finger_tip.y
                distance, _, img = self.findDistance((x1, y1), (x2, y2), img)
                self.volume.SetMasterVolumeLevelScalar(distance, None)
        
        return img

     def findDistance(self, p1, p2, img = None, color = (255, 0, 255), scale = 5):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = ((x1 + x2) / 2), ((y1 + y2) / 2)
        distance = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (int(x1 * img.shape[1]), int(y1 * img.shape[0])), scale, color, cv2.FILLED)
            cv2.circle(img, (int(x2 * img.shape[1]), int(y2 * img.shape[0])), scale, color, cv2.FILLED)
            cv2.circle(img, (int(cx * img.shape[1]), int(cy * img.shape[0])), scale, color, cv2.FILLED)
            cv2.line(img, (int(x1 * img.shape[1]), int(y1 * img.shape[0])), (int(x2 * img.shape[1]), int(y2 * img.shape[0])), color, max(1, scale // 3))
            
        return distance, info, img
    
def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('C:\\Users\\Rishavkumar\\Videos\\Captures\\output.mp4', fourcc, 20.0, (640, 480))
    
    while True:
        success, img = cap.read()
        if(success):
            img = detector.findHands(img)
            cv2.putText(img, 'Volume: ' + str(int(detector.volume.GetMasterVolumeLevelScalar() * 100)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            out.write(img)
            cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('c'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()