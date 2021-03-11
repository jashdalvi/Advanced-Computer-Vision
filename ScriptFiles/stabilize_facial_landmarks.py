import numpy as np
import cv2
import dlib

cap = cv2.VideoCapture(0)

landmark_detector_path = "../data/models/shape_predictor_68_face_landmarks.dat"

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor(landmark_detector_path)

def landmarks_to_points(landmarks):
    points = []
    for i in range(len(landmarks.parts())):
        points.append([landmarks.part(i).x,landmarks.part(i).y])

    return points 

frame_num = 0
show_stabilized = False

ret ,frame = cap.read()
prev_frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

points_detected = []
while True:
    start_time = cv2.getTickCount()

    ret,frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]
    width_resize_scale = frame_width/320
    height_resize_scale = frame_height/240

    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame_small_rgb = cv2.resize(frame_rgb,(320,240),interpolation = cv2.INTER_LINEAR)

    face_rects = face_detector(frame_small_rgb,0)
    if len(face_rects) == 0:
        continue

    for i in range(len(face_rects)):

        new_rect = dlib.rectangle(int(face_rects[i].left()*width_resize_scale),
                                int(face_rects[i].top()*height_resize_scale),
                                int(face_rects[i].right()*width_resize_scale),
                                int(face_rects[i].bottom()*height_resize_scale))

        landmarks = landmark_detector(frame_rgb,new_rect)

        points_detected = landmarks_to_points(landmarks)


    if frame_num == 0:
        points_detected_prev = points_detected
    
    points_detected_curr = points_detected

    points_detected_prev = np.array(points_detected_prev,dtype = np.float32).reshape(-1,2)
    points_detected_curr = np.array(points_detected_curr,dtype = np.float32).reshape(-1,2)

    eye_distance = int(cv2.norm(points_detected_curr[36] - points_detected_curr[45]))
    sigma = eye_distance * eye_distance / 400
    s = 2*int(eye_distance/4)+1

    #  Set up optical flow params
    lk_params = dict(winSize  = (s, s), maxLevel = 5, criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03))

    points_detected_op ,status , _ = cv2.calcOpticalFlowPyrLK(prev_frame_gray,frame_gray,points_detected_prev,None,**lk_params)

    points_detected_op = np.array(points_detected_op,dtype = np.float32).reshape(68,2)

    points = np.zeros_like(points_detected_op)

    for i in range(len(points_detected_prev)):
        d = cv2.norm(points_detected_prev[i] - points_detected_curr[i])
        alpha = np.exp(-d*d/sigma)  
        points[i] = (1 - alpha)*(points_detected_curr[i]) + alpha*(points_detected_op[i])


    if show_stabilized == True:
        for i in range(len(points)):
            cv2.circle(frame,(int(points[i][0]),int(points[i][1])),3,(255,0,0),-1)
    else:
        for i in range(len(points)):
            cv2.circle(frame,(int(points_detected_curr[i][0]),int(points_detected_curr[i][1])),3,(0,255,0),-1)


    fps = cv2.getTickFrequency()/(cv2.getTickCount() - start_time)
    cv2.putText(frame,"FPS : {:.3f}".format(fps),(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2,cv2.LINE_AA)
    points_detected_prev = points.copy()
    prev_frame_gray = frame_gray.copy()
    frame_num += 1
    cv2.imshow("Frame",frame)
    k = cv2.waitKey(25)
    if k == ord('q'):
        break
    elif k == ord('s'):
        show_stabilized = True
    elif k == ord('n'):
        show_stabilized = False
    


cap.release()
cv2.destroyAllWindows()