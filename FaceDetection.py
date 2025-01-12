import cv2

#* Loads a pre-trained face detection model from the Haar Cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#*  Opens the webcam for capturing video.
webcam = cv2.VideoCapture(0)
while True:
    #* webcam.read() -> Captures a frame from the webcam and img -> actual image captured
    successful_frame_read,img = webcam.read()

    #* Converts the colored frame (img) from BGR (blue, green, red) to grayscale
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #* Detects faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray_img,1.3,5)

    for (x,y,w,h) in faces:
        #* Draws a rectangle around the detected face
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

    #* Displays the resulting image
    cv2.imshow("Face detection",img)
    key = cv2.waitKey(10)
    if key==27:
        break
webcam.release()
cv2.destroyAllWindows()

