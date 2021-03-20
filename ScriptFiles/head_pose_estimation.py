import cv2
import numpy as np
import dlib
import time

# 3D Model Points of selected landmarks in an 
# arbitrary frame of reference
def get3dModelPoints():
  modelPoints = [[0.0, 0.0, 0.0],
                 [0.0, -330.0, -65.0],
                 [-225.0, 170.0, -135.0],
                 [225.0, 170.0, -135.0],
                 [-150.0, -150.0, -125.0],
                 [150.0, -150.0, -125.0]]
  return np.array(modelPoints, dtype=np.float64)


# 2D landmark points from all landmarks
def get2dImagePoints(shape):
  imagePoints = [[shape.part(30).x, shape.part(30).y],
                 [shape.part(8).x, shape.part(8).y],
                 [shape.part(36).x, shape.part(36).y],
                 [shape.part(45).x, shape.part(45).y],
                 [shape.part(48).x, shape.part(48).y],
                 [shape.part(54).x, shape.part(54).y]]
  return np.array(imagePoints, dtype=np.float64)


# Camera Matrix from focal length and focal center
def getCameraMatrix(focalLength, center):
  cameraMatrix = [[focalLength, 0, center[0]],
                  [0, focalLength, center[1]],
                  [0, 0, 1]]
  return np.array(cameraMatrix, dtype=np.float64)

def draw_polylines(image,start,end,closed = False):
    points = []
    for i in range(start,end +1):
        point = [landmarks.part(i).x,landmarks.part(i).y]
        points.append(point)

    points = np.array(points,dtype = np.int32).reshape(len(points),1,2)

    cv2.polylines(image,[points],closed, (0,255,0), thickness=2, lineType=cv2.LINE_8)



def draw_landmarks(image,landmarks):

    draw_polylines(image,0,16)              #jawline
    draw_polylines(image,17,21)             #lefteye-brow    
    draw_polylines(image,22,26)             #righteye-brow
    draw_polylines(image,36,41,True)        #left-eye
    draw_polylines(image,42,47,True)        #right-eye
    draw_polylines(image,27,30)             #nose-line
    draw_polylines(image,30,35,True)        #nose
    draw_polylines(image,48,59,True)        #outer-lip
    draw_polylines(image,60,67,True)        #inner-lip

def draw_points(image,landmarks):
    for i in range(len(landmarks.parts())):
        point = (landmarks.part(i).x,landmarks.part(i).y)
        cv2.circle(image,point,2,(0,255,0),-1)

landmark_detector_path = "../data/models/shape_predictor_68_face_landmarks.dat"

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor(landmark_detector_path)
cap = cv2.VideoCapture(0)
j = 0

while cap.isOpened():
    start_time = cv2.getTickCount()
    start_time_lib = time.time()

    ret,frame = cap.read()

    

    if not ret:
        break
    frame = cv2.resize(frame,(320,240),interpolation = cv2.INTER_LINEAR)
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    face_rects = face_detector(frame_rgb,0)

    if len(face_rects) == 0:
        continue

    for i in range(len(face_rects)):
        rect = dlib.rectangle(int(face_rects[i].left()),
                            int(face_rects[i].top()),
                            int(face_rects[i].right()),
                            int(face_rects[i].bottom()))

        landmarks = landmark_detector(frame_rgb,rect)

        draw_landmarks(frame,landmarks)

        model_points_3d = get3dModelPoints()
        image_points_2d = get2dImagePoints(landmarks)

        focalLength = frame.shape[1]
        center = (frame.shape[1]//2,frame.shape[0]//2)
        camera_matrix = getCameraMatrix(focalLength, center)
        dist_coeffs = np.zeros((4,1),dtype=np.float64)

        retval,rvec,tvec = cv2.solvePnP(model_points_3d,image_points_2d,camera_matrix,dist_coeffs)

        nose_point_3d = np.float64([[0.0,0.0,1000.0]])
        nose_points_2d , jacobian = cv2.projectPoints(nose_point_3d,rvec,tvec,camera_matrix,dist_coeffs)

        pt1 = (int(image_points_2d[0,0]),int(image_points_2d[0,1]))
        pt2 = (int(nose_points_2d[0,0,0]),int(nose_points_2d[0,0,1]))

        cv2.line(frame,pt1,pt2,(0,0,255),2)
    fps_lib = 1/(time.time() - start_time_lib)
    fps = cv2.getTickFrequency()/(cv2.getTickCount() - start_time)
    cv2.putText(frame,"FPS : {:.3f}".format(fps),(10,25),cv2.FONT_HERSHEY_SIMPLEX,
    0.7,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow("Landmark Detection",frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()