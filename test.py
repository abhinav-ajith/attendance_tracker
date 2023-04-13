import os
import cv2
cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPatheyes = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"

faceCascade = cv2.CascadeClassifier(cascPathface)
eyeCascade = cv2.CascadeClassifier(cascPatheyes)

count =0 

directory = '../train/image_data'
 
# iterate over files in
# that directory

import pandas as pd

df = pd.read_csv('../train/train.csv')

correct = 0
total =0
for filename in os.listdir(directory):
    if total >100:
        break
    f = os.path.join(directory, filename)
    video_capture = cv2.VideoCapture(f)
    ret,frame = video_capture.read()
    original_frame = cv2.flip(frame,2).copy()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # print(f)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        count +=1
        
        h= int(h*1.5)
        cv2.rectangle(frame, (x, y), (x + w, y + h),(0,0,0), -1)
        # faceROI = frame[y:y+h,x:x+w]

    frame = cv2.flip(frame,2)
    cv2.imshow('frame', frame)
    expected_count = df.loc[df['Name']==(f[-9:]),'HeadCount']
    if(len(expected_count)!=0):
        total+=1
        print(f'Person count is {count} while expected is {expected_count.iloc[0]}')
        if count == expected_count.iloc[0]:
            correct +=1
            print(f)
            cv2.imwrite(f'result_{f[-9:-4]}.jpg', frame)
            cv2.imwrite(f'original_{f[-9:-4]}.jpg', original_frame)
            # cv2.imwrite(f'result.jpg', frame)
            # cv2.imwrite(f'original.jpg', original_frame)
            # break
    count =0
    # cv2.imwrite('result.jpg', frame)
    pressedKey = cv2.waitKey(1) & 0xFF

    # enter 's' to pause while viewing video 
    if pressedKey == ord('s'):
        pressedKey = cv2.waitKey() & 0xFF
        if pressedKey == ord('q'):
            break

        
    # enter 'q' to exit while viewing video 
    if pressedKey == ord('q'):
        break

print(f'Accuracy : {float((correct/total)*100)}%')
# After the loop release the cap object
video_capture.release()
# Destroy all the windows
cv2.destroyAllWindows()