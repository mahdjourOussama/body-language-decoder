import numpy as np
import cv2
import mediapipe as mp
import pandas as pd
import pickle

mp_drawing=mp.solutions.drawing_utils
mp_holistic= mp.solutions.holistic

#initiate drawing Spec
hand=mp_drawing.DrawingSpec(color=(200,0,50),thickness=2,circle_radius=2)
head=mp_drawing.DrawingSpec(color=(50,200,100),thickness=1,circle_radius=1)
body=mp_drawing.DrawingSpec(color=(0,50,200),thickness=2,circle_radius=2)
joins_connections=mp_drawing.DrawingSpec(color=(119,88,19),thickness=2,circle_radius=2)

#openning and setting the webcam 
cap=cv2.VideoCapture(0)
cap.set(3,800) #setting the width of the frame
cap.set(4,700) #setting the height of the frame
cap.set(10,100) #setting the brightnesse of the frame

#initiate the mode of logistic regression
with open('D:/Project/AI/ML/body-language-decoder/body_language.pkl', 'rb') as f:
    model = pickle.load(f)

#initiate holistic mediapipe
with mp_holistic.Holistic(min_detection_confidence=0.6,min_tracking_confidence=0.6) as holistic:

    #live stream webcam feed
    while cap.isOpened():
        status,frame =cap.read() #capture the frame + status

        # Recolor Feed to RGB
        image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # Make detection
        result=holistic.process(image)
     
        
        # Recolor back to BRG
        image.flags.writeable = True
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        # making prediction about body language
        try:
            #grapping pose landmarks
            pose=result.pose_landmarks.landmark
            pose_row=list(np.array([[landmark.x,landmark.y,landmark.z,landmark.visibility] for landmark in pose]).flatten())

            #grapping face landmarks
            face=result.face_landmarks.landmark
            face_row=list(np.array([[landmark.x,landmark.y,landmark.z,landmark.visibility] for landmark in face]).flatten())
            row=face_row+pose_row
            x=pd.DataFrame([row])
            body_language_class=model.predict(x)[0]
            body_language_proba=model.predict_proba(x)[0]
            # Grab ear coords
            coords = tuple(np.multiply(
                            np.array(
                                (result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                 result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                        , [640,480]).astype(int))
            
            cv2.rectangle(image, 
                          (coords[0], coords[1]+20), 
                          (coords[0]+len(body_language_class)*20, coords[1]-30), 
                          (245, 117, 16), -1)
            cv2.putText(image, body_language_class, coords, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Get status box
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            
            # Display Class
            cv2.putText(image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0]
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display Probability
            cv2.putText(image, 'PROB'
                        , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_proba[np.argmax(body_language_proba)],2))
                        , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        except:
            pass
        

        
        cv2.imshow('WebCam',image) #stream the frame thought a window
        c = cv2.waitKey(1)
        if c == 27:#exit fucntion exit using esc
            break 
cap.release()
cv2.destroyAllWindows()