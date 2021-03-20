import numpy as np
import cv2
import dlib

def barrel_distort(src,k):
    height,width = src.shape[:2]

    x,y = np.meshgrid(np.arange(width),np.arange(height))

    x_norm = np.float32(x)/width - 0.5
    y_norm = np.float32(y)/height - 0.5

    r = np.sqrt(x_norm**2 + y_norm**2)

    dr = k*r*np.cos(np.pi*r)

    dr[r > 0.5] = 0

    rn = r - dr

    xd = cv2.divide((x_norm * rn),r)
    yd = cv2.divide((y_norm * rn),r)

    xd = (xd + 0.5)*width
    yd = (yd + 0.5)*height
    

    dst = cv2.remap(src,xd,yd,cv2.INTER_CUBIC)

    return dst



def landmarks_to_points(landmarks):
    points = []
    for i in range(len(landmarks.parts())):
        points.append([landmarks.part(i).x,landmarks.part(i).y])

    return points 

cap = cv2.VideoCapture(0)

face_detector = dlib.get_frontal_face_detector()
landmark_detector_path = "../data/models/shape_predictor_68_face_landmarks.dat"

landmark_detector = dlib.shape_predictor(landmark_detector_path)

while cap.isOpened():

    ret,frame = cap.read()

    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    face_rects = face_detector(frame_rgb,0)

    if len(face_rects) == 0:
        continue

    for i in range(len(face_rects)):
        new_rect = dlib.rectangle(int(face_rects[0].left()), 
                          int(face_rects[0].top()), 
                          int(face_rects[0].right()), 
                          int(face_rects[0].bottom()))

        landmarks = landmark_detector(frame_rgb,new_rect)

        points = landmarks_to_points(landmarks)

        radius = 30

        x_left = points[37][0] - radius
        y_left = points[37][1] - radius
        w_left = points[40][0] - points[37][0] + 2*radius
        h_left = points[41][1] - points[37][1] + 2*radius
        eye_roi_left = frame[y_left:y_left+h_left,x_left:x_left+w_left]

        x_right = points[43][0] - radius
        y_right = points[43][1] - radius
        w_right = points[46][0] - points[43][0] + 2*radius
        h_right = points[47][1] - points[43][1] + 2*radius
        eye_roi_right = frame[y_right:y_right+h_right,x_right:x_right+w_right]

        output = frame.copy()

        output_roi_left = barrel_distort(eye_roi_left,0.5)
        output_roi_right = barrel_distort(eye_roi_right,0.5)

        output[y_left:y_left+h_left,x_left:x_left+w_left] = output_roi_left
        output[y_right:y_right+h_right,x_right:x_right+w_right] = output_roi_right

    cv2.imshow("Bug eyes",output)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()





