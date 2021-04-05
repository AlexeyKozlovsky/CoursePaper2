import cv2


cam = cv2.VideoCapture(cv2.CAP_V4L2)
mouth_cascade = cv2.CascadeClassifier('Cascades/Mouth.xml')


while True:
    ret, img = cam.read()
    if not ret:
        print('Failed..')
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mouth = mouth_cascade.detectMultiScale(gray, 1.4, 20)
    

    croped_img = img
    for (x, y, w, h) in mouth:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
        croped_img = img[y:y+h,x:x+w]

    if croped_img is not None:
        cv2.imshow('croped_img', croped_img)
    cv2.imshow('img', img)
    k = cv2.waitKey(1)

    if k % 256 == 27:
        print('Closed..')

        break


