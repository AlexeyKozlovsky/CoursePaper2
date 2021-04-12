import cv2
import imutils
import dlib
import time
import numpy as np

class Chunck:
    def __init__(self, video_path, shape_predictor_path, size=(300, 270)):
        self.path = video_path
        self.size = size
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_path)
        self.x1_avg_face, self.x2_avg_face = 0, 0
        self.y1_avg_face, self.y2_avg_face = 0, 0
        self.x1_avg_landmark, self.x2_avg_landmark = 0, 0
        self.y1_avg_landmark, self.y2_avg_landmark = 0, 0
        self.angle = 0
        
        cap = cv2.VideoCapture(video_path)
        self.WIDTH = int(cap.get(3))
        self.HEIGHT = int(cap.get(4))
        self.FRAME_COUNT = cap.get(cv2.CAP_PROP_FRAME_COUNT)
     
    
    def prepare(self):
        """
        Метод для подготовки нахождения координат лица, лендмарков
        и вращения, чтобы губы были параллельно оси Ox
        """
        
        cap = cv2.VideoCapture(self.path)
        detections = 0
        self.x1_avg_face, self.x2_avg_face = 0, 0
        self.y1_avg_face, self.y2_avg_face = 0, 0
        self.x1_avg_landmark, self.x2_avg_landmark = 0, 0
        self.y1_avg_landmark, self.y2_avg_landmark = 0, 0
        self.x_max_landmark, self.y_max_landmark = 0, 0
        self.x_min_landmark, self.y_min_landmark = self.WIDTH, self.HEIGHT
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray)
                   
            for face in faces:
                detections += 1
                self.x1_avg_face += face.left()
                self.x2_avg_face += face.right()
                self.y1_avg_face += face.top()
                self.y2_avg_face += face.bottom()
                
                landmarks = self.predictor(gray, face)
                mouth_landmarks = landmarks.parts()[48:60]
                
                x_min, x_max, y_min, y_max = self.WIDTH, 0, self.HEIGHT, 0
                for i, landmark in enumerate(mouth_landmarks):
                    x, y = landmark.x, landmark.y
                    if i == 0:
                        self.x1_avg_landmark += x
                        self.y1_avg_landmark += y
                    elif i == 6:
                        self.x2_avg_landmark += x
                        self.y2_avg_landmark += y
                        
                    x_min, x_max = min(x, x_min), max(x, x_max)
                    y_min, y_max = min(y, y_min), max(y, y_max)
                    
                self.x_min_landmark = min(self.x_min_landmark, x_min)
                self.x_max_landmark = max(self.x_max_landmark, x_max)
                self.y_min_landmark = min(self.y_min_landmark, y_min)
                self.y_max_landmark = max(self.y_max_landmark, y_max)
                        
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        
        cap.release()
        
        self.x1_avg_face = int(self.x1_avg_face / detections)
        self.x2_avg_face = int(self.x2_avg_face / detections)
        self.y1_avg_face = int(self.y1_avg_face / detections)
        self.y2_avg_face = int(self.y2_avg_face / detections)
        
        self.x1_avg_landmark = int(self.x1_avg_landmark / detections)
        self.x2_avg_landmark = int(self.x2_avg_landmark / detections)
        self.y1_avg_landmark = int(self.y1_avg_landmark / detections)
        self.y2_avg_landmark = int(self.y2_avg_landmark / detections)
        
        a = self.x2_avg_landmark - self.x1_avg_landmark
        b = self.y2_avg_landmark - self.y1_avg_landmark
        self.angle = 180 * np.arctan2(b, a) / np.pi
        
        self.face = dlib.rectangle(self.x1_avg_face, self.y1_avg_face,
                                  self.x2_avg_face, self.y2_avg_face)
        
        self.mouth = dlib.rectangle(self.x_min_landmark, self.y_min_landmark, 
                                   self.x_max_landmark, self.y_max_landmark)
        
        
    def show(self, time_sleep=0):
        cap = cv2.VideoCapture(self.path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            blank = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
            
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            landmarks = self.predictor(gray, self.face)
            for i, landmark in enumerate(landmarks.parts()[48:60]):
                if i != 0:
                    x_prev, y_prev = x, y
                    cv2.line(blank, (x_prev, y_prev), (x, y), (255, 255, 255), 1)
                    
                x, y = landmark.x, landmark.y
                cv2.circle(frame, (x, y), 2, (0, 0, 255))
                
                #cv2.circle(blank, (x, y), 1, (255, 255, 255))
                
            cv2.imshow('blank line', blank)
                
                
            frame = imutils.rotate(frame, self.angle, center=(self.face.center().x,
                                                        self.face.center().y))
            blank = imutils.rotate(blank, self.angle, center=(self.face.center().x,
                                                        self.face.center().y))
               
            # Crop mouth area
            top = int(self.mouth.top() - 0.5 * self.mouth.height())
            bottom = int(self.mouth.bottom() + 0.5 * self.mouth.height())
            left = int(self.mouth.left() - 0.2 * self.mouth.width())
            right = int(self.mouth.right() + 0.2 * self.mouth.width())
            frame = frame[top : bottom, left : right]
            blank = blank[top : bottom, left : right]
            
            frame = imutils.resize(frame, width=self.size[0], height=self.size[1])
            blank = imutils.resize(blank, width=self.size[0], height=self.size[1])
            
            cv2.imshow('chunk', frame)
            cv2.imshow('blank', blank)
            
            time.sleep(time_sleep)
            
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    def to_file(self, filename):
        cap = cv2.VideoCapture(self.path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, 10, (self.size[0],
                                                    self.size[1]))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            blank = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
             
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            landmarks = self.predictor(gray, self.face)
            for i, landmark in enumerate(landmarks.parts()[48:60]):
                if i != 0:
                    x_prev, y_prev = x, y
                    cv2.line(blank, (x_prev, y_prev), (x, y), (255, 255, 255), 1)
                    
                x, y = landmark.x, landmark.y
                cv2.circle(frame, (x, y), 2, (0, 0, 255))
                
            frame = imutils.rotate(frame, self.angle, center=(self.face.center().x,
                                                        self.face.center().y))
            blank = imutils.rotate(blank, self.angle, center=(self.face.center().x,
                                                        self.face.center().y))
               
            # Crop mouth area
            top = int(self.mouth.top() - 0.5 * self.mouth.height())
            bottom = int(self.mouth.bottom() + 0.5 * self.mouth.height())
            left = int(self.mouth.left() - 0.2 * self.mouth.width())
            right = int(self.mouth.right() + 0.2 * self.mouth.width())
            frame = frame[top : bottom, left : right]
            blank = blank[top : bottom, left : right]
            
            frame = imutils.resize(frame, width=self.size[0], height=self.size[1])
            blank = imutils.resize(blank, width=self.size[0], height=self.size[1])
            blank = cv2.resize(blank, self.size)
            
            out.write(blank)
        
        cap.release()