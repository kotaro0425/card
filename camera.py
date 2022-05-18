from string import capwords
import cv2
import numpy
face_cascade_path = 'haarcascade_frontalface_default.xml'
eye_cascade_path = 'haarcascade_eye.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
cap = cv2.VideoCapture(0)
img = cv2.imread("Unknown-2")
card=cv2.resize(img,(200,100))


while True:
    ret, src = cap.read()
    
    cv2.putText(src, "RakutenCardMan!!", (50,100), cv2.FONT_HERSHEY_SIMPLEX,4,(0, 100,255), 3,cv2.LINE_AA)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(src_gray)
    for x, y, w, h in faces:
        face = src[y: y + h, x: x + w]
        face_gray = src_gray[y: y + h, x: x + w]
        eyes = eye_cascade.detectMultiScale(src_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            height, width = card.shape[:2]
            src[ey:height+ey,ex:width+ex]=card
    cv2.imshow('src',src)
    k = cv2.waitKey(1)
    if k == 27:
        break
    cv2.imshow('src',src)
# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()