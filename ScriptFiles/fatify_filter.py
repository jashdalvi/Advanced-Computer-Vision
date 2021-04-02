import cv2
import dlib
import numpy as np
from mls import MLSWarpImage

def add_boundary_points(arr,width,height):

    arr.append((0,0))
    arr.append((width//2,0))
    arr.append((width,0))
    arr.append((width,height//2))
    arr.append((width,height))
    arr.append((width//2,height))
    arr.append((0,height))
    arr.append((0,height//2))

    return arr
def landmarks_to_points(landmarks):
    points = []
    for i in range(len(landmarks.parts())):
        points.append((landmarks.part(i).x,landmarks.part(i).y))

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
    
    x1,y1,x2,y2 = int(face_rects[0].left()), int(face_rects[0].top()), int(face_rects[0].right()), int(face_rects[0].bottom())
    new_rect = dlib.rectangle(int(face_rects[0].left()), 
                          int(face_rects[0].top()), 
                          int(face_rects[0].right()), 
                          int(face_rects[0].bottom()))

    landmarks = landmark_detector(frame_rgb,new_rect)

    points = landmarks_to_points(landmarks)

    src_points = []
    dst_points = []

    width = int(x2-x1)
    height = int(y2-y1)
    src_points = add_boundary_points(src_points,width,height)
    dst_points = add_boundary_points(dst_points,width,height)

    frame_copy = frame.copy()
    face_roi = frame[y1:y2,x1:x2]

    anchorPoints = [1, 15, 30]

    # Points that will be deformed
    deformedPoints = [ 5, 6, 8, 10, 11]

    centerx , centery = points[30][0],points[30][1]
    for idx in anchorPoints:
        src_points.append(points[idx])
        dst_points.append(points[idx])

    for idx in deformedPoints:
        src_points.append(points[idx])
        dst_points.append((int(1.5*(points[idx][0] - centerx)+centerx),int(1.5*(points[idx][1] - centery)+centery)))

    src_points = np.array(src_points)
    dst_points = np.array(dst_points)

    output = MLSWarpImage(frame,src_points,dst_points)
    cv2.imshow("Filter",output)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





    